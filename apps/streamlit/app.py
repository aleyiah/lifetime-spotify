import io
import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

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
        st.write(f"Type: **dict**  |  Keys: **{len(json_obj.keys())}**")
        st.json(dict(list(json_obj.items())[:10]))
    else:
        st.write(f"Type: **{type(json_obj).__name__}**")
        st.json(json_obj)

    st.subheader("Merge all listening data")

    merge_now = st.button("Concatenate streaming history files")

    if merge_now:
        with st.spinner("Validating JSON files and merging listening events..."):
            df_events, meta = load_all_streaming_history(extract_dir, json_rel_paths)

        if df_events.empty:
            st.error("No streaming history events found across JSON files.")
            st.write("A few files we skipped (for debugging):")
            st.json(meta["skipped_files"][:10])
            st.stop()

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

        st.subheader("Merged dataset preview")
        st.dataframe(df_events.head(50), use_container_width=True)

        st.subheader("Quick sanity checks")
        st.write(f"Unique tracks: **{df_events['track_name'].nunique(dropna=True):,}**")
        st.write(f"Unique artists: **{df_events['artist_name'].nunique(dropna=True):,}**")
        st.write(f"Total minutes played: **{(df_events['ms_played'].sum() / 1000 / 60):,.1f}**")

