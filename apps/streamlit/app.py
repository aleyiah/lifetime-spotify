import io
import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st


# =========================
# Spotify streaming validators
# =========================

SCHEMA_A_REQUIRED = {"msPlayed"}  # older
SCHEMA_A_HINTS = {"endTime", "trackName", "artistName"}

SCHEMA_B_REQUIRED = {"ms_played"}  # extended
SCHEMA_B_HINTS = {
    "ts",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
}


def looks_like_event_dict(d: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (is_event, schema_name) for a single dict."""
    keys = set(d.keys())

    if SCHEMA_A_REQUIRED.issubset(keys) and len(keys.intersection(SCHEMA_A_HINTS)) >= 1:
        return True, "schema_a"

    if SCHEMA_B_REQUIRED.issubset(keys) and len(keys.intersection(SCHEMA_B_HINTS)) >= 1:
        return True, "schema_b"

    return False, "unknown"


def validate_streaming_history_json(json_obj: Any) -> Dict[str, Any]:
    """Validate that a loaded JSON object contains listening events."""
    result = {
        "is_streaming_history": False,
        "schema": None,
        "events_found": 0,
        "sample_event": None,
        "reason": "",
    }

    if not isinstance(json_obj, list):
        result["reason"] = f"Top-level JSON is {type(json_obj).__name__}, expected a list of events."
        return result

    if len(json_obj) == 0:
        result["reason"] = "JSON list is empty."
        return result

    n = min(200, len(json_obj))
    items = json_obj[:n]

    dict_items = [x for x in items if isinstance(x, dict)]
    if not dict_items:
        result["reason"] = "List items are not dicts; expected dict events."
        return result

    schema_counts = {"schema_a": 0, "schema_b": 0, "unknown": 0}
    first_event = None

    for d in dict_items:
        ok, schema = looks_like_event_dict(d)
        schema_counts[schema] += 1
        if ok and first_event is None:
            first_event = d

    eventish = schema_counts["schema_a"] + schema_counts["schema_b"]
    if eventish / len(dict_items) < 0.5:
        result["reason"] = f"Only {eventish}/{len(dict_items)} inspected items look like streaming events."
        return result

    schema = "schema_a" if schema_counts["schema_a"] >= schema_counts["schema_b"] else "schema_b"

    result["is_streaming_history"] = True
    result["schema"] = schema
    result["events_found"] = len(json_obj)
    result["sample_event"] = first_event
    result["reason"] = f"Validated as streaming history ({schema})."
    return result


# =========================
# ZIP + JSON helpers
# =========================

def is_zip_bytes(data: bytes) -> bool:
    return zipfile.is_zipfile(io.BytesIO(data))


def safe_extract_zip(zip_path: str, extract_to: str) -> None:
    """Safely extract a ZIP (prevents Zip Slip / path traversal)."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = os.path.normpath(member.filename)
            if member_path.startswith(("/", "\\")) or ".." in Path(member_path).parts:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(extract_to)


def find_json_files(root_dir: str) -> list[str]:
    """Return relative paths of all .json files under root_dir."""
    root = Path(root_dir)
    json_paths = sorted([p for p in root.rglob("*.json") if p.is_file()])
    return [str(p.relative_to(root)) for p in json_paths]


def load_json_safely(path: Path):
    """Load JSON; return (data, error)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


# =========================
# Merge / concatenate streaming history
# =========================

def normalize_events(json_obj: Any, schema: str) -> list[dict]:
    """
    Convert Schema A or B raw Spotify events into a unified column schema.
    Output columns (pre-clean): ts, ms_played, track_name, artist_name, album_name, source_schema
    """
    rows: list[dict] = []

    if not isinstance(json_obj, list):
        return rows

    if schema == "schema_a":
        for e in json_obj:
            if not isinstance(e, dict):
                continue
            rows.append(
                {
                    "ts": e.get("endTime"),
                    "ms_played": e.get("msPlayed"),
                    "track_name": e.get("trackName"),
                    "artist_name": e.get("artistName"),
                    "album_name": None,
                    "source_schema": "schema_a",
                }
            )

    elif schema == "schema_b":
        for e in json_obj:
            if not isinstance(e, dict):
                continue
            rows.append(
                {
                    "ts": e.get("ts"),
                    "ms_played": e.get("ms_played"),
                    "track_name": e.get("master_metadata_track_name"),
                    "artist_name": e.get("master_metadata_album_artist_name"),
                    "album_name": e.get("master_metadata_album_album_name"),
                    "source_schema": "schema_b",
                }
            )

    return rows


def load_all_streaming_history(extract_dir: Path, json_rel_paths: list[str]) -> tuple[pd.DataFrame, dict]:
    """Validate, load, normalize, and concatenate streaming history across all JSON files."""
    meta = {
        "streaming_files_used": [],
        "skipped_files": [],
        "total_events": 0,
        "schemas": {"schema_a": 0, "schema_b": 0},
    }

    all_rows: list[dict] = []

    for rel in json_rel_paths:
        path = extract_dir / rel
        json_obj, err = load_json_safely(path)
        if err:
            meta["skipped_files"].append({"file": rel, "reason": f"json parse error: {err}"})
            continue

        validation = validate_streaming_history_json(json_obj)
        if not validation["is_streaming_history"]:
            meta["skipped_files"].append({"file": rel, "reason": validation["reason"]})
            continue

        schema = validation["schema"]
        meta["schemas"][schema] += 1
        meta["streaming_files_used"].append(rel)

        all_rows.extend(normalize_events(json_obj, schema))

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce")

        # Parse + normalize timestamps to UTC
        df["ts_parsed"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

        # Drop invalid rows, rename canonical timestamp, remove raw ts
        df = (
            df.dropna(subset=["ms_played", "ts_parsed"])
              .rename(columns={"ts_parsed": "ts_utc"})
              .drop(columns=["ts"], errors="ignore")
              .reset_index(drop=True)
        )

        meta["total_events"] = len(df)

        # Privacy: drop any IP-related or sensitive network fields if present
        sensitive_cols = [
            "ip_addr",
            "ip_address",
            "client_ip",
            "conn_ip",
            "network_ip",
        ]
        df = df.drop(columns=[c for c in sensitive_cols if c in df.columns])

        # Enforce canonical schema
        expected_cols = {
            "ts_utc",
            "ms_played",
            "track_name",
            "artist_name",
            "album_name",
            "source_schema",
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

    return df, meta


# =========================
# Era helpers (month-granular, UTC)
# =========================

def month_start_utc(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, month, 1, tzinfo=timezone.utc))


def next_month_start_utc(year: int, month: int) -> pd.Timestamp:
    if month == 12:
        return pd.Timestamp(datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    return pd.Timestamp(datetime(year, month + 1, 1, tzinfo=timezone.utc))


def ranges_overlap(a_start: pd.Timestamp, a_end_excl: pd.Timestamp,
                   b_start: pd.Timestamp, b_end_excl: pd.Timestamp) -> bool:
    # Half-open intervals: [start, end)
    return (a_start < b_end_excl) and (b_start < a_end_excl)


def apply_era_filter_month(df: pd.DataFrame, start_year: int, start_month: int, end_year: int, end_month: int) -> pd.DataFrame:
    start_ts = month_start_utc(start_year, start_month)
    end_exclusive = next_month_start_utc(end_year, end_month)
    return df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] < end_exclusive)].copy()


def safe_rerun() -> None:
    # Works across Streamlit versions
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Lifetime Spotify", page_icon="🎧", layout="centered")

st.title("🎧 Lifetime Spotify")
st.write("Upload your Spotify data export as a **.zip** file.")

uploaded = st.file_uploader(
    "Choose a ZIP file",
    type=["zip"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a ZIP to begin.")
    st.stop()

data = uploaded.getvalue()

if not is_zip_bytes(data):
    st.error("That file isn't a valid ZIP. Please upload a .zip export.")
    st.stop()

max_mb = 50
if len(data) > max_mb * 1024 * 1024:
    st.error(f"File too large. Max allowed is {max_mb}MB.")
    st.stop()

st.success(f"Uploaded: {uploaded.name} ({len(data)/1024/1024:.2f} MB)")

with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = Path(tmpdir) / "upload.zip"
    zip_path.write_bytes(data)

    extract_dir = Path(tmpdir) / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        safe_extract_zip(str(zip_path), str(extract_dir))
    except Exception as e:
        st.error(f"Couldn't extract ZIP: {e}")
        st.stop()

    json_rel_paths = find_json_files(str(extract_dir))

    st.subheader("JSON files found")
    if not json_rel_paths:
        st.warning("No JSON files found in the ZIP. Make sure you uploaded the Spotify data export ZIP.")
        st.stop()

    st.write(f"Found **{len(json_rel_paths)}** JSON file(s). Non-JSON files are ignored.")

    with st.expander("Show JSON file list"):
        st.write(json_rel_paths)

    selected_rel = st.selectbox("Preview a JSON file", json_rel_paths)
    selected_path = extract_dir / selected_rel

    json_obj, err = load_json_safely(selected_path)
    if err:
        st.error(f"Couldn't parse JSON: {selected_rel}\n\n{err}")
        st.stop()

    validation = validate_streaming_history_json(json_obj)

    st.subheader("Streaming History Validation")
    if validation["is_streaming_history"]:
        st.success(validation["reason"])
        st.write(f"Events in file: **{validation['events_found']}**")
        st.write(f"Detected schema: **{validation['schema']}**")
        st.caption("Sample event (first validated event):")
        st.json(validation["sample_event"])
    else:
        st.error("This JSON does not look like Spotify streaming history.")
        st.write(validation["reason"])
        st.caption("Tip: choose another JSON file—many Spotify export files aren't listening events.")

    st.subheader("Preview")
    if isinstance(json_obj, list):
        st.write(f"Type: **list**  |  Length: **{len(json_obj)}**")
        st.json(json_obj[:2])
    elif isinstance(json_obj, dict):
        st.write(f"Type: **dict**  |  Keys: **{len(json_obj)}**")
        st.json(dict(list(json_obj.items())[:10]))
    else:
        st.write(f"Type: **{type(json_obj).__name__}**")
        st.json(json_obj)

    # =====================================================
    # NEW: Merge all listening data (cached in session state)
    # =====================================================

    st.subheader("Merge all listening data")

    # Initialize session state containers
    if "df_events" not in st.session_state:
        st.session_state.df_events = None
    if "meta" not in st.session_state:
        st.session_state.meta = None
    if "merged" not in st.session_state:
        st.session_state.merged = False
    if "eras" not in st.session_state:
        st.session_state.eras = []

    merge_now = st.button("Concatenate streaming history files")

    # When clicked, compute once and store results
    if merge_now:
        with st.spinner("Validating JSON files and merging listening events..."):
            df_events, meta = load_all_streaming_history(extract_dir, json_rel_paths)

        if df_events.empty:
            st.session_state.df_events = None
            st.session_state.meta = meta
            st.session_state.merged = False

            st.error("No streaming history events found across JSON files.")
            st.write("A few files we skipped (for debugging):")
            st.json(meta["skipped_files"][:10])
            st.stop()

        st.session_state.df_events = df_events
        st.session_state.meta = meta
        st.session_state.merged = True

    # Render merged UI if we have results in session_state
    if st.session_state.merged and st.session_state.df_events is not None:
        df_events = st.session_state.df_events
        meta = st.session_state.meta

        st.success("Merged listening data!")

        # 🔒 Privacy notice
        st.info(
            "🔒 **Privacy notice**: We automatically removed IP addresses and other "
            "network identifiers from your Spotify data before processing. "
            "This helps protect your privacy, and no sensitive location or "
            "network information is stored or displayed."
        )

        with st.expander("What data was removed?"):
            st.write(
                "We removed IP addresses and related network identifiers (such as client "
                "or connection IP fields). These fields are not required for listening "
                "insights and are excluded to protect your privacy."
            )

        st.write(f"Streaming history files used: **{len(meta['streaming_files_used'])}**")
        st.write(f"Detected schemas: **A={meta['schemas']['schema_a']}**, **B={meta['schemas']['schema_b']}**")
        st.write(f"Total listening events: **{meta['total_events']:,}**")

        with st.expander("Show streaming files used"):
            st.write(meta["streaming_files_used"])

        # =========================================
        # Eras (dropdown months, no overlaps)
        # =========================================
        st.subheader("Define eras of your life")

        current_year = date.today().year
        years = list(range(2011, current_year + 1))
        months = list(range(1, 13))
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_map = {i + 1: m for i, m in enumerate(month_labels)}

        with st.expander("Add a new era", expanded=True):
            era_name = st.text_input("Era name", placeholder="e.g., Middle school", key="era_name")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                start_year = st.selectbox("Start year", years, index=0, key="era_start_year")
            with c2:
                start_month = st.selectbox(
                    "Start month",
                    months,
                    format_func=lambda m: month_map[m],
                    index=8,  # Sep
                    key="era_start_month",
                )
            with c3:
                end_year = st.selectbox("End year", years, index=min(2, len(years) - 1), key="era_end_year")
            with c4:
                end_month = st.selectbox(
                    "End month",
                    months,
                    format_func=lambda m: month_map[m],
                    index=5,  # Jun
                    key="era_end_month",
                )

            add_era = st.button("Add era", key="add_era_btn")

            if add_era:
                name = era_name.strip()
                if not name:
                    st.error("Please enter an era name.")
                else:
                    start_ts = month_start_utc(int(start_year), int(start_month))
                    end_exclusive = next_month_start_utc(int(end_year), int(end_month))

                    if end_exclusive <= start_ts:
                        st.error("End must be the same as or after the start month.")
                    else:
                        overlap_with = None
                        for e in st.session_state.eras:
                            e_start = month_start_utc(e["start_year"], e["start_month"])
                            e_end_excl = next_month_start_utc(e["end_year"], e["end_month"])
                            if ranges_overlap(start_ts, end_exclusive, e_start, e_end_excl):
                                overlap_with = e["name"]
                                break

                        if overlap_with:
                            st.error(
                                f"This era overlaps with **{overlap_with}**. "
                                "Please adjust dates so eras do not overlap."
                            )
                        else:
                            st.session_state.eras.append(
                                {
                                    "name": name,
                                    "start_year": int(start_year),
                                    "start_month": int(start_month),
                                    "end_year": int(end_year),
                                    "end_month": int(end_month),
                                }
                            )
                            st.success(
                                f"Added era: {name} "
                                f"({int(start_year)}-{int(start_month):02d} → {int(end_year)}-{int(end_month):02d})"
                            )

        if st.session_state.eras:
            eras_df = pd.DataFrame(st.session_state.eras).copy()
            eras_df["_start"] = eras_df.apply(lambda r: month_start_utc(r["start_year"], r["start_month"]), axis=1)
            eras_df = eras_df.sort_values("_start").drop(columns=["_start"]).reset_index(drop=True)

            eras_df["start"] = eras_df.apply(lambda r: f"{r['start_year']}-{r['start_month']:02d}", axis=1)
            eras_df["end"] = eras_df.apply(lambda r: f"{r['end_year']}-{r['end_month']:02d}", axis=1)
            eras_df = eras_df[["name", "start", "end"]]

            st.write("Your eras:")
            st.dataframe(eras_df, use_container_width=True)

            col_a, col_b = st.columns([2, 1])
            with col_a:
                era_options = ["All time"] + eras_df["name"].tolist()
                selected_era_name = st.selectbox("Select an era", era_options, key="selected_era")
            with col_b:
                if st.button("Clear eras", key="clear_eras_btn"):
                    st.session_state.eras = []
                    safe_rerun()

            df_view = df_events
            if selected_era_name != "All time":
                era = next(e for e in st.session_state.eras if e["name"] == selected_era_name)
                df_view = apply_era_filter_month(
                    df_events,
                    era["start_year"],
                    era["start_month"],
                    era["end_year"],
                    era["end_month"],
                )
                st.info(
                    f"Filtering to **{selected_era_name}** "
                    f"({era['start_year']}-{era['start_month']:02d} → {era['end_year']}-{era['end_month']:02d})"
                )

            st.subheader("Dataset preview (filtered)")
            st.dataframe(df_view.head(50), use_container_width=True)

            st.subheader("Quick sanity checks (filtered)")
            st.write(f"Events: **{len(df_view):,}**")
            st.write(f"Unique tracks: **{df_view['track_name'].nunique(dropna=True):,}**")
            st.write(f"Unique artists: **{df_view['artist_name'].nunique(dropna=True):,}**")
            st.write(f"Total minutes played: **{(df_view['ms_played'].sum() / 1000 / 60):,.1f}**")
        else:
            st.caption("Add at least one era to enable filtering by era.")

        st.subheader("Quick sanity checks (all time)")
        st.write(f"Unique tracks: **{df_events['track_name'].nunique(dropna=True):,}**")
        st.write(f"Unique artists: **{df_events['artist_name'].nunique(dropna=True):,}**")
        st.write(f"Total minutes played: **{(df_events['ms_played'].sum() / 1000 / 60):,.1f}**")
