import io
import os
import json
import zipfile
import tempfile
import hashlib
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


def zip_hash(data: bytes) -> str:
    """Stable identifier for the uploaded ZIP bytes."""
    return hashlib.sha256(data).hexdigest()


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

def load_all_streaming_history(extract_dir: Path, json_rel_paths: list[str], zip_hash_id: str = None) -> tuple[pd.DataFrame, dict]:
    """Validate, load, normalize, and concatenate streaming history across all JSON files.
    
    Note: Not cached globally. Results are stored in st.session_state only.
    This avoids keeping large dataframes in memory across sessions.
    """
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
        df["ts_parsed"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

        df = (
            df.dropna(subset=["ms_played", "ts_parsed"])
              .rename(columns={"ts_parsed": "ts_utc"})
              .drop(columns=["ts"], errors="ignore")
              .reset_index(drop=True)
        )

        meta["total_events"] = len(df)

        sensitive_cols = ["ip_addr", "ip_address", "client_ip", "conn_ip", "network_ip"]
        df = df.drop(columns=[c for c in sensitive_cols if c in df.columns])

        expected_cols = {"ts_utc", "ms_played", "track_name", "artist_name", "album_name", "source_schema"}
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


def get_era_for_timestamp(ts: pd.Timestamp, eras: list[dict]) -> str:
    """Find which era a timestamp falls into. Returns era name or 'Pre-era' if no match."""
    if pd.isna(ts) or not eras:
        return "Pre-era"
    
    for era in eras:
        era_start = month_start_utc(era["start_year"], era["start_month"])
        era_end = next_month_start_utc(era["end_year"], era["end_month"])
        if era_start <= ts < era_end:
            return era["name"]
    
    return "Pre-era"


def apply_era_filter_month(df: pd.DataFrame, start_year: int, start_month: int, end_year: int, end_month: int) -> pd.DataFrame:
    start_ts = month_start_utc(start_year, start_month)
    end_exclusive = next_month_start_utc(end_year, end_month)
    return df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] < end_exclusive)].copy()


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================
# Era persistence (ZIP-scoped)
# =========================

ERAS_BY_ZIP_FILE = Path(".streamlit/eras_by_zip.json")
LEGACY_ERAS_FILE = Path(".streamlit/eras.json")  # optional migration from Option 1


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_eras_for_zip(zip_id: str) -> list[dict]:
    """
    Returns eras saved for this zip_id.
    Also performs a one-time migration from legacy .streamlit/eras.json if present.
    """
    data = _read_json_file(ERAS_BY_ZIP_FILE)
    if isinstance(data, dict) and isinstance(data.get(zip_id), list):
        return data[zip_id]

    # One-time migration: if no eras_by_zip entry yet, but legacy file exists, copy it in.
    legacy = _read_json_file(LEGACY_ERAS_FILE)
    if isinstance(legacy, list) and legacy:
        save_eras_for_zip(zip_id, legacy)
        return legacy

    return []


def save_eras_for_zip(zip_id: str, eras: list[dict]) -> None:
    ERAS_BY_ZIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json_file(ERAS_BY_ZIP_FILE)
    if not isinstance(data, dict):
        data = {}
    data[zip_id] = eras
    ERAS_BY_ZIP_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_eras_for_zip(zip_id: str) -> None:
    data = _read_json_file(ERAS_BY_ZIP_FILE)
    if not isinstance(data, dict):
        return
    data[zip_id] = []
    ERAS_BY_ZIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    ERAS_BY_ZIP_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
current_zip_id = zip_hash(data)

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

    if not json_rel_paths:
        st.error("❌ No JSON files found in the ZIP. Make sure you uploaded the Spotify data export ZIP.")
        st.stop()

    # Quick validation check on first JSON file
    first_json_path = extract_dir / json_rel_paths[0]
    first_json_obj, err = load_json_safely(first_json_path)
    if err or not first_json_obj:
        st.error("❌ Could not parse JSON files. Please check your export.")
        st.stop()
    
    validation = validate_streaming_history_json(first_json_obj)
    if not validation["is_streaming_history"]:
        st.error("❌ JSON files do not appear to be Spotify streaming history.")
        st.stop()
    
    st.success(f"✅ Found and validated {len(json_rel_paths)} streaming history file(s)")

    # =========================================
    # Merge all listening data (cached)
    # =========================================

    st.subheader("Merge all listening data")

    if "df_events" not in st.session_state:
        st.session_state.df_events = None
    if "meta" not in st.session_state:
        st.session_state.meta = None
    if "merged" not in st.session_state:
        st.session_state.merged = False

    # Track zip id in session (if upload changes, reload eras and reset merge state)
    if "zip_id" not in st.session_state or st.session_state.zip_id != current_zip_id:
        st.session_state.zip_id = current_zip_id
        st.session_state.eras = load_eras_for_zip(current_zip_id)
        st.session_state.selected_era = "All time"
        # If you want to keep merged data when the same zip is re-uploaded, we leave it.
        # But if the zip changes, clear merge cache:
        st.session_state.df_events = None
        st.session_state.meta = None
        st.session_state.merged = False

    merge_now = st.button("Concatenate streaming history files")

    if merge_now:
        with st.spinner("Loading and merging streaming history..."):
            df_events, meta = load_all_streaming_history(extract_dir, json_rel_paths)

        if df_events.empty:
            st.session_state.df_events = None
            st.session_state.meta = meta
            st.session_state.merged = False
            st.error("❌ No streaming history events found. Check your data export.")
            st.stop()

        st.session_state.df_events = df_events
        st.session_state.meta = meta
        st.session_state.merged = True

    if st.session_state.merged and st.session_state.df_events is not None:
        df_events = st.session_state.df_events
        meta = st.session_state.meta
        zip_id = st.session_state.zip_id

        st.success("Merged listening data!")

        # 🔒 Privacy notice (single)
        st.info(
            "🔒 **Privacy notice**: We automatically removed IP addresses and other "
            "network identifiers from your Spotify data before processing. "
            "No sensitive location or network information is stored or displayed."
        )

        st.write(f"✅ Merged {len(meta['streaming_files_used'])} file(s) with **{meta['total_events']:,}** events")

        # -----------------------------
        # Era selection (ZIP-scoped persisted eras)
        # -----------------------------
        eras = st.session_state.eras or []
        df_view = df_events
        selected_era_name = "All time"
        # ...existing code...
        st.subheader("Sanity checks")
        st.write(f"View: **{selected_era_name}**")
        st.write(f"Events: **{len(df_view):,}**")
        st.write(f"Unique tracks: **{df_view['track_name'].nunique(dropna=True):,}**")
        st.write(f"Unique artists: **{df_view['artist_name'].nunique(dropna=True):,}**")
        st.write(f"Total minutes played: **{(df_view['ms_played'].sum() / 1000 / 60):,.1f}**")

        # =========================================
        # Era creation / management (NOW IMMEDIATELY AFTER SANITY CHECKS)
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
                            save_eras_for_zip(zip_id, st.session_state.eras)
                            st.success(
                                f"Added era: {name} "
                                f"({int(start_year)}-{int(start_month):02d} → {int(end_year)}-{int(end_month):02d})"
                            )
                            safe_rerun()

        if st.session_state.eras:
            eras_df = pd.DataFrame(st.session_state.eras).copy()
            eras_df["_start"] = eras_df.apply(lambda r: month_start_utc(r["start_year"], r["start_month"]), axis=1)
            eras_df = eras_df.sort_values("_start").drop(columns=["_start"]).reset_index(drop=True)
            eras_df["start"] = eras_df.apply(lambda r: f"{r['start_year']}-{r['start_month']:02d}", axis=1)
            eras_df["end"] = eras_df.apply(lambda r: f"{r['end_year']}-{r['end_month']:02d}", axis=1)
            eras_df = eras_df[["name", "start", "end"]]

            st.caption("Saved eras (scoped to this upload):")
            col_title, col_clear = st.columns([3, 1])
            with col_clear:
                if st.button("Clear eras", key="clear_eras_btn_era_section"):
                    st.session_state.eras = []
                    clear_eras_for_zip(zip_id)
                    safe_rerun()
            
            # Display eras with individual delete buttons
            for idx, era in enumerate(st.session_state.eras):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(era["name"])
                with col2:
                    st.caption(f"{era['start_year']}-{era['start_month']:02d} → {era['end_year']}-{era['end_month']:02d}")
                with col3:
                    if st.button("Delete", key=f"delete_era_{idx}"):
                        st.session_state.eras.pop(idx)
                        clear_eras_for_zip(zip_id)
                        safe_rerun()

            # ====================================
            # ERA-SPECIFIC STATISTICS
            # ====================================
            st.divider()
            st.header("📊 ERA-SPECIFIC STATISTICS")
            st.subheader("Top Artists and Tracks by Era")
            if st.session_state.df_events is not None:
                df_events = st.session_state.df_events
                era_options = eras_df["name"].tolist()
                selected_era = st.selectbox("Select an era to view stats:", era_options, key="era_stats_select")
                st.info("💡 Choose an era from the dropdown above to explore your listening patterns during that time period.")
                era_row = eras_df[eras_df["name"] == selected_era].iloc[0]
                start_year, start_month = map(int, era_row["start"].split("-"))
                end_year, end_month = map(int, era_row["end"].split("-"))
                df_era = apply_era_filter_month(df_events, start_year, start_month, end_year, end_month)
                st.caption(f"{start_year}-{start_month:02d} → {end_year}-{end_month:02d} | Events: {len(df_era):,}")
                if df_era.empty:
                    st.info("No data for this era.")
                else:
                    # Top artists
                    top_artists_era = (
                        df_era.groupby("artist_name")["ms_played"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(5)
                        .reset_index()
                    )
                    top_artists_era["minutes_played"] = (top_artists_era["ms_played"] / 1000 / 60).round(1)
                    st.markdown("**Top 5 Artists**")
                    st.dataframe(top_artists_era[["artist_name", "minutes_played"]], use_container_width=True, hide_index=True)
                    # Top tracks
                    top_tracks_era = (
                        df_era.groupby(["track_name", "artist_name"])["ms_played"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(5)
                        .reset_index()
                    )
                    top_tracks_era["minutes_played"] = (top_tracks_era["ms_played"] / 1000 / 60).round(1)
                    st.markdown("**Top 5 Tracks**")
                    st.dataframe(top_tracks_era[["track_name", "artist_name", "minutes_played"]], use_container_width=True, hide_index=True)

            # ====================================
            # ALL-TIME STATISTICS
            # ====================================
            st.divider()
            st.header("🎵 ALL-TIME STATISTICS")
            st.subheader("Top Artists and Tracks")

            # Top Artists by total playtime
            top_artists = (
                df_view.groupby("artist_name")["ms_played"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top_artists["minutes_played"] = (top_artists["ms_played"] / 1000 / 60).round(1)

            st.markdown("**Top 10 Artists by Minutes Played**")
            st.dataframe(top_artists[["artist_name", "minutes_played"]], use_container_width=True, hide_index=True)

            # Top Tracks by total playtime
            top_tracks = (
                df_view.groupby(["track_name", "artist_name"])["ms_played"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top_tracks["minutes_played"] = (top_tracks["ms_played"] / 1000 / 60).round(1)

            st.markdown("**Top 10 Tracks by Minutes Played**")
            st.dataframe(top_tracks[["track_name", "artist_name", "minutes_played"]], use_container_width=True, hide_index=True)

            st.subheader("Listening Trends: Top 10 Artists Over Time")
            if not top_artists.empty:
                top10_artists = top_artists["artist_name"].tolist()
                df_top = df_events[df_events["artist_name"].isin(top10_artists)].copy()
                # Create bi-yearly period column
                df_top["year"] = df_top["ts_utc"].dt.year
                df_top["month"] = df_top["ts_utc"].dt.month
                df_top["half"] = df_top["month"].apply(lambda m: 1 if m <= 6 else 2)
                df_top["bi_yearly"] = df_top["year"].astype(str) + " H" + df_top["half"].astype(str)

                # Pivot table: rows=artist, columns=bi-yearly, values=minutes played
                pivot = pd.pivot_table(
                    df_top,
                    index="artist_name",
                    columns="bi_yearly",
                    values="ms_played",
                    aggfunc="sum",
                    fill_value=0,
                )
                # Convert ms_played to minutes
                pivot = (pivot / 1000 / 60).round(1)
                # Add total column for sorting
                pivot["Total"] = pivot.sum(axis=1)
                pivot = pivot.sort_values("Total", ascending=False).drop(columns=["Total"])

                st.markdown(
                    "This line graph shows the minutes played for each top artist in every half-year period. "
                    "You can see when artists appear, peak, or drop off in your listening history."
                )
                # Prepare long-form DataFrame for plotting
                pivot_reset = pivot.reset_index()
                pivot_melt = pivot_reset.melt(id_vars="artist_name", var_name="bi_yearly", value_name="minutes_played")
                # Sort bi_yearly periods chronologically
                period_order = sorted(pivot.columns, key=lambda x: (int(x.split()[0]), int(x.split()[1][1:])) if x != "Total" else (9999, 0))
                pivot_melt = pivot_melt[pivot_melt["bi_yearly"].isin(period_order)]
                pivot_melt["bi_yearly"] = pd.Categorical(pivot_melt["bi_yearly"], categories=period_order, ordered=True)
                pivot_melt = pivot_melt.sort_values(["artist_name", "bi_yearly"])

                import altair as alt
                # Interactive selection for artist
                selection = alt.selection_single(fields=["artist_name"], bind="legend", name="Select")
                chart = alt.Chart(pivot_melt).mark_line(point=True).encode(
                    x=alt.X("bi_yearly:N", title="Half-Year Period"),
                    y=alt.Y("minutes_played:Q", title="Minutes Played"),
                    color=alt.Color("artist_name:N", title="Artist"),
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
                    tooltip=["artist_name", "bi_yearly", "minutes_played"]
                ).add_selection(
                    selection
                ).properties(
                    width=800,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)

                # Show top album and track for selected artist and each bi-yearly period
                artist_options = ["(None)"] + top10_artists
                artist_choice = st.selectbox("Highlight artist for details:", artist_options, index=0, key="selected_artist_for_biyearly")
                if artist_choice and artist_choice != "(None)":
                    df_artist = df_top[df_top["artist_name"] == artist_choice].copy()
                    if not df_artist.empty:
                        st.markdown(f"#### Top Album and Track for {artist_choice} by Half-Year")
                        try:
                            periods = sorted(df_artist["bi_yearly"].unique(), key=lambda x: (int(x.split()[0]), int(x.split()[1][1:])))
                        except Exception:
                            periods = []
                        rows = []
                        for period in periods:
                            df_period = df_artist[df_artist["bi_yearly"] == period]
                            # Top album
                            if "album_name" in df_period.columns and not df_period["album_name"].isnull().all():
                                top_album = (
                                    df_period.groupby("album_name")["ms_played"].sum().sort_values(ascending=False).head(1)
                                )
                                album_name = top_album.index[0] if not top_album.empty else None
                                album_minutes = (top_album.iloc[0] / 1000 / 60) if not top_album.empty else 0
                            else:
                                album_name = None
                                album_minutes = 0
                            # Top track
                            top_track = (
                                df_period.groupby("track_name")["ms_played"].sum().sort_values(ascending=False).head(1)
                            )
                            track_name = top_track.index[0] if not top_track.empty else None
                            track_minutes = (top_track.iloc[0] / 1000 / 60) if not top_track.empty else 0
                            rows.append({
                                "Half-Year": period,
                                "Top Album": album_name,
                                "Album Minutes": round(album_minutes, 1),
                                "Top Track": track_name,
                                "Track Minutes": round(track_minutes, 1),
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("No data for this artist in any half-year period.")
                    else:
                        st.info("No data for this artist.")
                else:
                    st.info("💡 Select an artist to see top album and track details.")

            # ====================================
            # FIRST LISTEN DATE FOR TOP ARTISTS
            # ====================================
            st.divider()
            st.subheader("🎤 When Did You First Listen to These Artists?")

            # Get top 25 artists by total playtime
            top_25_artists = (
                df_view.groupby("artist_name")["ms_played"]
                .sum()
                .sort_values(ascending=False)
                .head(25)
                .reset_index()
            )

            # Create a mapping of artist info
            first_listen_data = {}
            artists_by_first_listen = []
            eras = st.session_state.eras or []
            
            for artist in top_25_artists["artist_name"]:
                artist_listens = df_view[df_view["artist_name"] == artist]
                first_listen = artist_listens["ts_utc"].min()
                total_minutes = (artist_listens["ms_played"].sum() / 1000 / 60)
                
                # Find when they started listening regularly (first day with 8+ plays)
                daily_plays = artist_listens.groupby(artist_listens["ts_utc"].dt.date).size()
                regular_listening_date = None
                for date, play_count in daily_plays.items():
                    if play_count >= 8:
                        regular_listening_date = pd.Timestamp(date, tz=timezone.utc)
                        break
                
                # Calculate days until regular listening
                days_to_regular = None
                if pd.notna(first_listen) and pd.notna(regular_listening_date):
                    days_to_regular = (regular_listening_date - first_listen).days
                
                # Get the era for first listen
                first_listen_era = get_era_for_timestamp(first_listen, eras)
                
                first_listen_data[artist] = {
                    "First Listen": first_listen.strftime("%B %d, %Y") if pd.notna(first_listen) else "Unknown",
                    "First Listen Date": first_listen,
                    "First Listen Era": first_listen_era,
                    "Regular Listening": regular_listening_date.strftime("%B %d, %Y") if pd.notna(regular_listening_date) else "Never",
                    "Regular Listening Date": regular_listening_date,
                    "Days to Regular": days_to_regular,
                    "Total Minutes": round(total_minutes, 1),
                }
                artists_by_first_listen.append((artist, total_minutes))
            
            # Sort by total minutes played (most to least)
            artists_by_first_listen.sort(key=lambda x: x[1], reverse=True)
            sorted_artists = [artist for artist, _ in artists_by_first_listen]

            st.info("💡 Select an artist to discover the first day you listened to them!")

            # Dropdown to select artist
            selected_artist = st.selectbox(
                "Choose an artist from your top 25:",
                sorted_artists,
                index=0,
                key="first_listen_artist_selector"
            )

            if selected_artist:
                info = first_listen_data[selected_artist]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("First Listen", info["First Listen"])
                    st.caption(f"Era: {info['First Listen Era']}")
                with col2:
                    st.metric("Started Regular Listening", info["Regular Listening"])
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    if info["Days to Regular"] is not None:
                        st.metric("Days to Big Fan", f"{info['Days to Regular']} days")
                    else:
                        st.metric("Days to Big Fan", "N/A")
                with col4:
                    st.metric("Total Minutes", f"{info['Total Minutes']:,.1f}")
                
                st.markdown(
                    "💡 This shows when you first discovered this artist and when you became a regular listener (8+ plays in a day)!"
                )
