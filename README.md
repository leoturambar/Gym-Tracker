# Gym Tracker

Personal training log with AI-powered workout analysis. Tracks sessions, 
computes muscle load over time, and generates narrative coaching via LLM.

## Features

- Log workouts with auto-populated exercise history
- Relative Training Volume (RTV) per muscle group
- Muscle radar chart with configurable time comparisons
- Bodyweight tracking
- AI coaching analysis (general, balance, progression, next session)

## Stack

Python · Streamlit · Pandas · Plotly · Anthropic API (claude-sonnet)

## Setup

```bash
git clone https://github.com/leoturambar/Gym-Tracker.git
cd Gym-Tracker

conda create -n gymtracker python=3.11
conda activate gymtracker
pip install -r requirements.txt

streamlit run app.py
```

**Anthropic API key** — set once as a system environment variable on Windows,
persists across all projects:
```bash
setx ANTHROPIC_API_KEY "sk-ant-..."
```
Reopen the terminal after running `setx`.

## Launch from desktop (Windows)

Double-click `launch.bat` — no terminal needed.

## Project structure

```
gym-tracker/
├── app.py           # Streamlit UI and tab layout
├── config.py        # Exercise library, muscle mapping
├── data_manager.py  # Session and bodyweight read/write
├── metrics.py       # RTV calculations, muscle scoring
├── llm.py           # Claude API calls and prompt building
├── launch.bat       # Windows launcher
└── data/
    └── sessions.csv # Your training data (not tracked in git)
```

## Data privacy

`data/sessions.csv` and `data/bodyweight.csv` are excluded from git 
via `.gitignore` — your personal data stays local.

## License

MIT — see [LICENSE](LICENSE)