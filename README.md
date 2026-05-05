# Gym Tracker

This started as a frustration with spreadsheets. I was tracking loads in Excel, manually computing whether I was getting stronger, and periodically realizing I had no idea whether my quad volume last month was actually more than the month before — or whether that meant anything relative to my bodyweight at the time. Gym Tracker is what replaced that. It's a Streamlit app that logs sessions, computes muscle load over time using a bodyweight-normalized metric I call RTV, and feeds the data to an LLM for coaching analysis. The interesting question wasn't "can I track workouts" — it was "what can a language model actually do when it has real, structured, personal data to work with?"

<!-- SCREENSHOT: app overview showing all three tabs -->

## Relative Training Volume

The core metric the app is built around is Relative Training Volume (RTV). Raw volume — kilograms lifted, sets completed — is almost meaningless without context. A 60 kg bench press means something very different to a 60 kg athlete than to a 90 kg one. Training is fundamentally relative to who is doing it.

RTV normalizes load by bodyweight and incorporates volume, producing a dimensionless number comparable across time and across people. The formula depends on exercise type. For a weighted exercise: `(load / bodyweight) × (reps / 10) × sets` — using 10 as the reference rep count. A standard 4×10 session at any load produces the same per-set RTV as a simple load/bodyweight ratio would, but a 4×15 leg extension scores 50% higher, and an AMRAP set where you grind out 14 reps scores proportionally more than a controlled 10. For pure bodyweight exercises, the load term is always 1.0 since you're always moving your own weight. For weighted bodyweight movements like pull-ups or dips — where you can add weight or use a band for assistance — the formula is `(bodyweight + added_weight) / bodyweight`, and the added weight can be negative when a band is helping. For timed exercises like planks, RTV is `duration / 120`, using 120 seconds as the reference.

Muscle contribution is fractional, not binary. Each exercise has a muscles dictionary mapping muscle group to a fractional weight summing to 1.0 — a chest press contributes 0.60 to chest, 0.20 to shoulders, 0.20 to triceps, not equally to all three. These weights are sourced from ExRx.net and stored in the exercise database.

Bodyweight measurements matter because they're in the denominator. `get_bodyweight_on()` in data_manager.py always retrieves the most recent measurement on or before a session's date — historical sessions are never retroactively recalculated with current weight. If you weighed 78 kg in February and 75 kg today, your February data reflects 78 kg.

Period comparison scores are normalized by calendar weeks (RTV per week), not session count. This makes all periods directly comparable regardless of length, and aligns with the reference athlete benchmark which is also expressed in weekly terms.

<!-- SCREENSHOT: Analisi tab showing radar chart and progression chart -->

## Training Intensity Model

Beyond basic load tracking, the app models different intensity strategies on the final set of each exercise:

**AMRAP** (As Many Reps As Possible) — the final set is taken to technical failure at the same weight as the working sets. The actual reps achieved are logged. The set count already includes the AMRAP set, so a logged set count of 4 means 3 straight sets plus 1 AMRAP. Applied to exercises like hip thrust, calf raise, leg curl, and bicep curls. The logged rep count feeds directly into RTV and into 1RM estimation.

**Drop inverse** — the final set uses an increased weight taken to failure, then immediately drops to a lower weight and continues to failure. Two additional data points are logged: the increased weight with its reps, and the drop weight with its reps. The set count includes the up-set; the drop set is stored for the LLM context but excluded from RTV. Applied to incline chest press, pec fly, tricep cables, leg extension, and bicep curls.

**Fixed plus** — the final set uses an increased weight but for a fixed, predetermined rep count. Not taken to failure. Applied to leg press variations.

**No modifier** — straight sets only. Some exercises are kept this way by choice (chest press, shoulder press, dips for shoulder health; leg press and leg extension for knee health; lower back extensions for spinal load management).

1RM is estimated on demand using the Epley formula: `load × (1 + reps / 30)`. For AMRAP sets where actual reps are logged, this gives a meaningful strength estimate without ever testing a true maximum.

## The App

The app has three tabs.

**Allenamento** is where sessions get entered and reviewed. At the top, you select the training day (Upper Push, Lower Quad, Upper Pull, or Lower Hip) and the date — all on one row alongside the rename and add-day buttons. Each exercise appears as a card with a consistent column layout: series count, reps, and weight on one line, with set type and final-set fields on the lines below — all vertically aligned so the grid never shifts. Timed exercises like planks show series and duration instead. Excluded warmup exercises don't appear. You can skip any exercise, remove cards from the session, and add extra exercises on the fly — including ones not yet in the database, which triggers an ExRx lookup to populate their muscle data. When you save a session, structural changes (exercises added or removed, set type changes) are automatically written back to exercises.json, keeping the exercise library in sync with how the program is actually being run. Below the log form, the full session history is accessible in the same tab: every past session in reverse chronological order, expandable to show all exercises with their loads.

<!-- SCREENSHOT: Allenamento tab showing exercise cards and session history -->

**Analisi** shows everything on one scrollable page. The top half has two panels side by side: the muscle balance radar on the left (a polar chart with eight axes — chest, shoulders, triceps, back, biceps, quads, hamstrings, core) and a bar chart on the right showing the same data numerically. You choose a comparison period (week, month, or year); the current period is compared to the previous same-length period, and an auto-computed reference athlete overlay provides a benchmark. All values are in RTV per week, making different period lengths directly comparable. The bottom half has two panels side by side: load in kg over time (left) and RTV over time (right) for a single exercise you select. Below the charts, the AI coaching section lets you run one of four analysis modes without leaving the tab.

<!-- SCREENSHOT: Analisi tab showing radar and bar chart above, progression charts below, AI section at bottom -->

**Profilo** manages the user context that feeds into AI analysis. Bodyweight is logged with dates (0.1 kg precision) and stored historically — the table shows every measurement and is used to look up the correct weight for any past session. A free-text persistent notes field carries injuries, chronic limitations, and long-term goals into every LLM prompt. A training goal selector covers hypertrophy, strength, muscular endurance, body recomposition, and maintenance. An exercise management section allows adding new exercises with LLM-assisted muscle group suggestion, and enriching existing exercises with muscle weights from ExRx.net by pasting the exercise URL.

## Exercise Database

The exercise library lives in `data/exercises.json` rather than hardcoded configuration. Each entry carries the exercise name, type, day assignments, muscle contribution weights, default set type, safety flags, a `reference_load` value, and ExRx URL if enriched. This means the library is a real data artifact — it grows as you add exercises, persists across code changes, and can be enriched with reference data without touching the codebase.

`reference_load` is the representative load (kg or seconds) used to auto-compute the reference athlete benchmark. The reference athlete is not hardcoded anywhere — `get_reference_rtv_weekly()` in metrics.py derives it at runtime from exercise loads at 75 kg bodyweight, 4 sets, 10 reps, weighted by how often each exercise appears across training days.

Muscle weights come from ExRx.net, the most comprehensive publicly available database of exercise biomechanics. The enrichment flow in the Profilo tab fetches the ExRx page for any exercise by URL, extracts primary and secondary muscle involvement using the LLM, and writes the result back to the database. Exercises enriched this way are flagged with `"source": "exrx"`.

## LLM Backend

The backend is configurable via the `LLM_BACKEND` flag in llm.py. Setting it to `"claude"` uses the Anthropic API with `claude-sonnet-4-20250514`. Setting it to `"ollama"` uses a local Ollama instance via its OpenAI-compatible endpoint at `http://localhost:11434/v1`, with `qwen2.5:14b` as the default model. The Ollama path uses the `openai` Python package as the HTTP client — it's not in `requirements.txt` because it's optional, install it separately if you want local inference. The prompt and response handling are identical for both backends.

The prompt ends with an instruction to always close with a Dodgeball quote. This is load-bearing for morale.

## Data Model

Sessions are stored flat in `data/sessions.csv`. Each row is one exercise from one session — sessions are identified by a Unix timestamp used as `session_id`. The schema carries: `session_id, date, day_id, day_name, exercise, variant, type, sets, reps, value, value2, set_type, reps_actual, value_drop, reps_drop, skipped, note`. Older rows missing newer columns are backfilled with safe defaults at load time. Bodyweight history is in `data/bodyweight.csv`, one measurement per date. Persistent profile notes are plain text in `data/memory.txt`. The exercise library is in `data/exercises.json`. All four are local files excluded from git.

## Setup

```bash
git clone https://github.com/leoturambar/Gym-Tracker.git
cd Gym-Tracker

conda create -n gymtracker python=3.11
conda activate gymtracker
pip install -r requirements.txt

streamlit run app.py
```

If you want to use the Claude backend, set your API key as a persistent environment variable on Windows:

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Reopen the terminal after running `setx`.

If you want to use Ollama instead, pull a model (`ollama pull qwen2.5:14b`) and make sure the Ollama server is running before launching the app. The `LLM_BACKEND` flag in `llm.py` controls which one is used.

On Windows, double-clicking `launch.bat` starts the app without a terminal.

## Project Structure

```
gym-tracker/
├── app.py              # Streamlit UI, three-tab layout
├── config.py           # Exercise library loader, muscle mapping
├── data_manager.py     # Session and bodyweight CSV read/write, schema defaults
├── metrics.py          # RTV computation, muscle scoring, period filtering, 1RM
├── llm.py              # Prompt building, LLM backend abstraction, ExRx extraction
├── launch.bat          # Windows one-click launcher
├── data/               # Local data (gitignored)
│   ├── sessions.csv    # Training log, one row per exercise per session
│   ├── bodyweight.csv  # Bodyweight history
│   ├── memory.txt      # Persistent profile notes
│   └── exercises.json  # Exercise library with muscle weights and metadata
└── assets/             # Screenshots
```

## License

MIT — see [LICENSE](LICENSE)
