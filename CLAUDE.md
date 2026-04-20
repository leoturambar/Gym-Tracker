# Gym Tracker — Claude Code Context

## What this is
Personal training log with AI-powered workout analysis. Tracks sessions,
computes Relative Training Volume (RTV) per muscle group, and generates
narrative coaching via LLM. Built as a portfolio project demonstrating
structured AI integration with real, personal data.

## Stack
Python · Streamlit · Pandas · Plotly · Anthropic API (claude-sonnet) · Ollama

## Project structure
- app.py — Streamlit UI, four-tab layout
- config.py — exercise library loader, muscle mapping, reference athlete
- data_manager.py — session and bodyweight read/write, schema defaults
- metrics.py — RTV calculations, muscle scoring, period filtering, 1RM estimation
- llm.py — Claude API calls, prompt building, session and muscle context formatting
- launch.bat — Windows one-click launcher
- data/ — personal training data (gitignored, never touch)
- data/exercises.json — exercise library with muscle weights, set types, metadata
- assets/ — screenshots for README

## Tab layout
1. Allenamento — session logging (card-based) + session history (Storico)
2. Analisi — muscle balance radar (top) + exercise progression charts (bottom)
3. AI — four LLM coaching modes
4. Profilo — bodyweight log, persistent notes, goal, exercise management tools

## Key logic

### RTV (Relative Training Volume)
The core metric. Implemented in metrics.py. Formula by exercise type:
- weighted: `(load / bodyweight) × (reps / 10) × sets`
- bodyweight: `1.0 × (reps / 10) × sets`
- weighted_bw: `((bodyweight + added_weight) / bodyweight) × (reps / 10) × sets`
- timed: `duration / 120` (120s reference, sets not applied)
- excluded: always 0.0

For AMRAP set type: uses `reps_actual` instead of `reps` if available.
For drop_inverse set type: computes RTV for base sets at base weight + one set at
increased weight (value2/reps_actual); value_drop/reps_drop are stored but not
included in the RTV calculation.

Muscle contribution is fractional, not binary. Each exercise has a `muscles`
dict in exercises.json mapping muscle group → weight (summing to 1.0).
RTV is multiplied by these weights when accumulating per-muscle scores.

Period comparison scores are normalized by session count (RTV per session),
not raw totals — this prevents calendar-length bias between periods.

Do not simplify or replace RTV without explicit instruction.

### Exercise library
Defined in data/exercises.json. Loaded at startup by config.py, which merges
with hardcoded fallback defaults. Schema per exercise:
- name, type, day_ids, muscles (dict), set_type, no_amrap, variants, default, source, exrx_url

`get_exercise_meta(name)` in config.py returns the full metadata dict for any
exercise. Used by app.py for UI pre-population and by metrics.py for fractional
RTV per muscle.

### Session data model
sessions.csv columns (one row per exercise per session):
`session_id, date, day_id, day_name, exercise, variant, type, sets, reps,
value, value2, set_type, reps_actual, value_drop, reps_drop, skipped, note`

Old rows missing new columns are backfilled with safe defaults by load_sessions().
Never add new columns without updating load_sessions() defaults.

### Set types
- standard — straight sets, no special final set
- amrap — final set to technical failure at same weight; logs reps_actual
- drop_inverse — final set at increased weight (value2) to failure, then
  immediate drop to lower weight (value_drop) to failure; logs reps_actual + reps_drop
- fixed_plus — final set at increased weight (value2), fixed reps, not to failure
- none — no intensity modifier (warmup exercises, timed holds)

### 1RM estimation
`estimate_1rm(load, reps)` in metrics.py uses Epley formula: `load × (1 + reps/30)`.
Standalone utility, not used inside RTV calculations.

### LLM backend
Configurable via `LLM_BACKEND` flag in llm.py.
- "claude" — Anthropic API, claude-sonnet-4-20250514, uses system parameter
- "ollama" — local Ollama via OpenAI-compatible endpoint, system message in messages array
Both backends receive the same system prompt and user message structure.
The prompt always ends with a Dodgeball quote. This is load-bearing for morale.

### AI coaching modes (llm.py)
Four focus modes, each with a full paragraph of instructions:
- general — patterns, consistency, RTV distribution vs. goal
- balance — structural vs. recent muscle imbalances, one corrective action
- progression — per-exercise trend analysis, AMRAP reps_actual as progression signal
- next_session — concrete next session plan with specific load suggestions

### ExRx enrichment
In the Profilo tab, users paste ExRx URLs for individual exercises. The app
fetches the page and calls the LLM to extract muscle weights into the exercises.json
muscles dict, setting source: "exrx". HTTP requests use browser-like headers
to avoid bot detection.

## Rules
- Write all code and docstrings in English
- Never modify or read anything in data/ — personal training data, gitignored
- Never modify config files that contain API keys or credentials
- Do not push sensitive data, keys, or personal preferences to git
- When adding features, follow existing module separation:
  UI logic → app.py, data I/O → data_manager.py, metrics → metrics.py, LLM → llm.py
- Never add a column to the session schema without updating load_sessions() defaults
- exercises.json is the source of truth for exercise metadata — config.py hardcoded
  dicts are fallback only; JSON wins on conflict
- Ask before restructuring the tab layout or card layout
