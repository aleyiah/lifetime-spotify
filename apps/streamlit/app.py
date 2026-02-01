import io
import os
import zipfile
import tempfile
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Lifetime Spotify", page_icon="🎧")

st.title("🎧 Lifetime Spotify")
st.write("Upload your Spotify data export as a **.zip** file.")

# ZIP-only uploader (UI-level filter)
uploaded = st.file_uploader(
    "Choose a ZIP file",
    type=["zip"],          # only allow .zip in the picker
    accept_multiple_files=False
)

def is_zip_bytes(data: bytes) -> bool:
    """Extra validation beyond file extension."""
    return zipfile.is_zipfile(io.BytesIO(data))

def safe_extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Safely extract a ZIP (prevents Zip Slip / path traversal).
    """
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = os.path.normpath(member.filename)

            # Block absolute paths and parent traversal
            if member_path.startswith(("/", "\\")) or ".." in Path(member_path).parts:
                raise ValueError(f"Unsafe path in zip: {member.filename}")

        zf.extractall(extract_to)

if uploaded is not None:
    # Read bytes once (Streamlit file object)
    data = uploaded.getvalue()

    # Hard validation: must actually be a zip
    if not is_zip_bytes(data):
        st.error("That file isn't a valid ZIP. Please upload a .zip Spotify export.")
        st.stop()

    # Optional: size limit (example: 50MB)
    max_mb = 50
    if len(data) > max_mb * 1024 * 1024:
        st.error(f"File too large. Max allowed is {max_mb}MB.")
        st.stop()

    st.success(f"Uploaded: {uploaded.name} ({len(data)/1024/1024:.2f} MB)")

    # Save + extract to a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(data)

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            safe_extract_zip(zip_path, extract_dir)
        except Exception as e:
            st.error(f"Couldn't extract ZIP: {e}")
            st.stop()

        # Show what we got
        extracted_files = []
        for root, _, files in os.walk(extract_dir):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), extract_dir)
                extracted_files.append(rel)

        st.subheader("Files inside ZIP")
        st.write(extracted_files if extracted_files else "No files found inside the ZIP.")
