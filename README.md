# Gym Tracker

This started as a frustration with spreadsheets. I was tracking loads in Excel, manually computing whether I was getting stronger, and periodically realizing I had no idea whether my quad volume last month was actually more than the month before — or whether that meant anything relative to my bodyweight at the time. Gym Tracker is what replaced that. It's a Streamlit app that logs sessions, computes muscle load over time using a bodyweight-normalized metric I call RTV, and feeds the data to an LLM for coaching analysis. The interesting question wasn't "can I track workouts" — it was "what can a language model actually do when it has real, structured, personal data to work with?"

<!-- SCREENSHOT: app overview showing all six tabs -->

## Relative Training Volume

The core metric the app is built around is Relative Training Volume (RTV). Raw volume — kilograms lifted, sets completed — is almost meaningless without context. A 60 kg bench press means something very different to a 60 kg athlete than to a 90 kg one. Training is fundamentally relative to who is doing it.

RTV normalizes load by bodyweight, producing a dimensionless number that is comparable across time and across people. The formula depends on exercise type. For a weighted exercise, RTV is simply `load / bodyweight` — a bench press of 60 kg at 75 kg bodyweight yields 0.80 RTV. For a pure bodyweight exercise (push-up, hanging leg raise), RTV is always 1.0, because the load is always the athlete's own weight. For weighted bodyweight movements like pull-ups or dips — where you can add weight or use a resistance band for assistance — the formula is `(bodyweight + added_weight) / bodyweight`, and the added weight can be negative when a band is helping. A 75 kg athlete doing assisted pull-ups with 15 kg of band support has an effective load of 60 kg, so RTV is `60 / 75 = 0.80`. For timed exercises like planks, RTV is `duration / 120`, using 120 seconds as the reference point.

This means bodyweight measurements matter. The app stores them with dates, and `get_bodyweight_on()` in [data_manager.py](data_manager.py) always retrieves the most recent measurement on or before a session's date — historical sessions are never retroactively recalculated with current weight. If you weighed 78 kg in February and 75 kg today, your February data reflects 78 kg. That matters for long-term trend analysis.

<!-- SCREENSHOT: progression tab showing both kg chart and RTV chart side by side -->

## The App

The app has six tabs. Here's what each one actually does.

**Log** is where sessions get entered. You pick one of four training days from the program (Upper 1 for chest and triceps, Lower 1 for quads and core, Upper 2 for back and biceps, Lower 2 for hamstrings and core), select the date, and enter loads. The form pre-populates each exercise with the last recorded value for that day type, so you only need to change what changed. Each exercise has a skip checkbox for days when you drop something. There's a free-text note field for the session — how you felt, what was different, any pain. These notes travel into the LLM context later.

**Storico** (History) shows every session in reverse chronological order. Each expands to show all exercises with their loads. Skipped exercises appear with strikethrough. Sessions can be deleted.

**Progressi** (Progression) tracks a single exercise over time. You select any weighted exercise and get four summary metrics — number of sessions logged, maximum load ever, most recent load, and total delta from first to last. Below that, two charts: load in kg over time and RTV over time. The RTV chart only appears if bodyweight data exists; otherwise there's a prompt to set it in the profile tab.

<!-- SCREENSHOT: exercise progression tab with kg and RTV charts for a specific exercise -->

**Radar** is the muscle balance view. It's a polar chart with eight axes — Petto (chest), Spalle (shoulders), Tricipiti, Quadricipiti, Core, Femorali (hamstrings), Schiena (back), Bicipiti — one for each muscle group in the muscle map. You choose between two metrics: RTV (the normalized load) or frequency (raw exercise counts). You also choose a comparison mode: all time vs. the planned training schema, the last 7 days vs. the preceding 7, the current calendar month vs. the previous, or the current year vs. the previous year.

When comparing against the schema, the reference polygon is computed directly from `config.py` — it counts how many exercises in the program target each muscle group, so it reflects what the planned split is supposed to produce. When comparing time periods, both polygons are computed from actual session data. There's an optional third overlay: a reference athlete, also defined in `config.py`, representing an approximately 75 kg athlete executing the full program with reasonable loads. The reference athlete values are scaled by a period multiplier so the comparison stays meaningful regardless of time window. Below the radar, horizontal progress bars show absolute values for the current period with two decimal places.

<!-- SCREENSHOT: radar chart in week vs previous week mode with reference athlete overlay visible -->

**Analisi AI** is where the LLM sits. More on this below.

**Profilo** (Profile) handles the user context that the AI reads. Bodyweight measurements are stored with dates and shown in a sortable table. There's a free-text persistent notes field — injuries, chronic limitations, long-term goals — that gets injected into every LLM prompt. There's also a training goal selector with five options: hypertrophy, strength, muscular endurance, body recomposition, and maintenance.

## AI Coaching

The AI analysis tab sends your training data to a language model framed as a strength and conditioning coach. There are four focus modes, selectable before generating.

**General analysis** asks the model to identify training patterns and overall progress across recent sessions. It sees session history, session notes, RTV values, and muscle load distribution.

**Muscular balance** directs the model specifically at the radar data — which muscle groups are over or undertrained relative to others, whether the imbalance is structural or just recent.

**Load progression** focuses on whether loads are trending consistently upward over time, or stagnating, or inconsistent. Given that RTV values are included per exercise in the session history, the model can see both absolute weight and relative-to-bodyweight trends.

**Next session** asks for a concrete recommendation for the next training session given everything it knows — recent fatigue signals from the notes, load history, and any limitations mentioned in the persistent profile.

All four modes receive the same context: the current bodyweight, the training goal, the persistent notes, the last 10 sessions with per-exercise RTV values and session notes, and a normalized muscle load distribution for the selected time period. When a period comparison is active, the model also receives the period-over-period delta for each muscle group, formatted as arrows (▲ / ▼ / =) with numeric deltas.

<!-- SCREENSHOT: AI analysis tab with a sample output from the progression focus mode -->

## LLM Backend

The backend is configurable via the `LLM_BACKEND` flag in [llm.py](llm.py). Setting it to `"claude"` uses the Anthropic API with `claude-sonnet-4-20250514`. Setting it to `"ollama"` uses a local Ollama instance via its OpenAI-compatible endpoint at `http://localhost:11434/v1`, with `qwen2.5:14b` as the default model. The Ollama path uses the `openai` Python package as the HTTP client — it's not in `requirements.txt` because it's optional, so install it separately if you want local inference. The prompt and response handling are identical for both backends; the difference is just the API call.

The prompt ends with an instruction to always close with a Dodgeball quote. This is load-bearing for morale.

## Data Model

Sessions are stored flat in `data/sessions.csv`. Each row is one exercise from one session — sessions are identified by a Unix timestamp used as `session_id`. Bodyweight history is in `data/bodyweight.csv`, one measurement per date. Persistent profile notes are plain text in `data/memory.txt`. All three are local files excluded from git. The exercise-to-muscle mapping in [config.py](config.py) is the lookup table that makes the radar and LLM muscle summaries work — it maps each exercise name to a list of muscle groups it targets.

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
├── app.py           # Streamlit UI, six-tab layout
├── config.py        # Exercise library, muscle mapping, reference athlete
├── data_manager.py  # Session and bodyweight CSV read/write, memory persistence
├── metrics.py       # RTV computation, muscle scoring, period filtering, progression
├── llm.py           # Prompt building, LLM backend abstraction
├── launch.bat       # Windows one-click launcher
├── data/            # Local data (gitignored)
└── assets/          # Screenshots
```

## License

MIT — see [LICENSE](LICENSE)
