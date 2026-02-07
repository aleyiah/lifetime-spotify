# 🎵 Lifetime Spotify: Your Personal Listening Story

> Beyond Spotify Wrapped: Define your own life eras and see how your music tells your story.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-green)](https://github.com/aleyiah/lifetime-spotify)

Part of [moonpath.dev](https://moonpath.dev) — a portfolio of data-driven storytelling projects.

---

## 📖 What is This?

**This is not Spotify Wrapped.** This is your complete listening history, analyzed on *your* terms.

Spotify gives you year-end summaries. We give you **life-era narratives**. Define custom periods like "College Years," "Post-Breakup Summer," or "NYC Era" — and see how your musical taste evolved through the moments that mattered.

### Key Features

✨ **Era-Based Analysis** — Not just years. Define periods that match your life story  
📊 **Lifetime Insights** — Top artists, tracks, and listening patterns across your entire Spotify history  
🎨 **Shareable Visualizations** — Export charts, summaries, and era cards as PDFs  
🔒 **Privacy First** — All processing happens locally. Your data stays yours.  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Your complete Spotify streaming history ([how to request it](https://support.spotify.com/us/article/data-rights-and-privacy-settings/))

### Installation

```bash
# Clone the repository
git clone https://github.com/aleyiah/lifetime-spotify.git
cd lifetime-spotify

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
lifetime-spotify/
├── apps/                   # Streamlit application code
├── packages/               # Core analysis modules
│   ├── ingestion/         # ZIP upload and JSON parsing
│   ├── validation/        # Schema validation (multiple Spotify formats)
│   ├── analysis/          # Era-based analytics engine
│   └── visualization/     # Chart generation and exports
├── docs/                   # Documentation and guides
├── requirements.txt        # Python dependencies
└── README.md              # You are here
```

---

## 🎯 How It Works

### 1. **Upload Your Data**

Request your Spotify data from your [account privacy settings](https://www.spotify.com/account/privacy/). You'll receive a ZIP file with JSON files containing your complete streaming history.

### 2. **Define Your Eras**

Create custom date ranges that match your life:
- "Freshman Year" (Sep 2020 - May 2021)
- "London Semester" (Jan 2022 - Jun 2022)  
- "Post-Grad Glow-Up" (Jul 2023 - Present)

### 3. **Get Insights**

For each era, see:
- **Top Artists & Tracks** — Your most-played music
- **Listening Time** — Hours spent in each era
- **Artist Diversity** — How varied your taste was
- **Behavioral Patterns** — When you listened most

### 4. **Export & Share**

Generate shareable PDFs with charts, stats, and optional "era vibe" summaries.

---

## 🛠️ Technical Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core analysis engine |
| **Pandas** | Data processing and transformation |
| **Streamlit** | Interactive web interface |
| **Plotly/Matplotlib** | Visualization and charting |
| **FastAPI** | (Planned) Backend API for scale |

### Data Validation

Handles multiple Spotify schema versions:
- Legacy streaming history format (2015-2021)
- Extended streaming history format (2021+)
- Validates required fields, normalizes timestamps, removes IP addresses

---

## 📊 What Makes This Different?

### vs. Spotify Wrapped
- **Custom eras** (not just calendar years)
- **Full history** (not just top 100 songs)
- **Your narrative** (define what periods matter)

### vs. Other Spotify Tools
- **Privacy-first** — All processing is local
- **Era-centric** — Designed around life chapters, not arbitrary time windows
- **Longitudinal** — See trends across your entire listening journey

---

## 🗺️ Roadmap

### ✅ Completed (Phase 1)
- [x] ZIP upload and JSON parsing
- [x] Multi-schema validation
- [x] Era definition UI
- [x] Basic analytics (top artists, tracks, listening time)
- [x] Timestamp normalization and privacy filtering

### 🔄 In Progress (Phase 2)
- [ ] Shareable visualizations (Canva exports)
- [ ] PDF export functionality
- [ ] Module selection toggles
- [ ] Loading indicators and error handling

### 🔮 Planned (Phase 3)
- [ ] Optional AI-generated "era vibe" summaries (Claude API)
- [ ] Tarot-style era cards (deterministic, playful)
- [ ] Public deployment (Hugging Face Spaces)
- [ ] FastAPI backend for scalability

See [PROJECT_PLAN.csv](PROJECT_PLAN.csv) for detailed timeline.

---

## 🤝 Contributing

This is a personal portfolio project, but feedback and suggestions are welcome!

**Found a bug?** Open an issue with:
- Your Spotify data format (legacy vs. extended)
- Steps to reproduce
- Error message or unexpected behavior

**Have an idea?** Share it in the discussions tab!

---

## 📜 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Spotify** for making listening history available
- **Streamlit** for the amazing web app framework
- **You** for caring about your music story

---

## 📬 Contact

Built by [Aleyiah Peña](https://moonpath.dev)

Questions? Reach out via [LinkedIn](https://linkedin.com/in/aleyiahpena) or open an issue.

---

**⚠️ Privacy Notice**

This app processes your Spotify data **locally on your machine**. No listening history is sent to external servers. Optional AI features (era summaries) only send anonymized, aggregated statistics — never raw listening events.

Your data. Your story. Your control.
