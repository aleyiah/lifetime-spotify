from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

import streamlit as st

st.set_page_config(page_title="Lifetime Spotify", page_icon="🎧", layout="centered")

st.title("🎧 Lifetime Spotify")
st.caption("First Streamlit checkpoint: app runs locally.")

st.write("Next: connect Spotify OAuth + show top tracks/artists.")
