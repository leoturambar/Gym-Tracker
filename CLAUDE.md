# Gym Tracker — Claude Code Context

## What this is
Personal training log with AI-powered workout analysis. Tracks sessions,
computes Relative Training Volume (RTV) per muscle group, and generates
narrative coaching via LLM. Built as a portfolio project demonstrating
structured AI integration with real, personal data.

## Stack
Python · Streamlit · Pandas · Plotly · Anthropic API (claude-sonnet) · Ollama

## Project structure
- app.py — Streamlit UI, three-tab layout
- config.py — exercise library loader, muscle mapping (no hardcoded reference athlete)
- data_manager.py — session and bodyweight read/write, schema defaults
- metrics.py — RTV calculations, muscle scoring, period filtering, 1RM estimation
- llm.py — Claude API calls, prompt building, session and muscle context formatting
- launch.bat — Windows one-click launcher
- data/ — personal training data (gitignored, never touch)
- data/exercises.json — exercise library with muscle weights, set types, metadata
- assets/ — screenshots for README

## Tab layout
1. Allenamento — session logging (card-based) + session history (Storico, paginated,
   full sets×reps / set-type detail per exercise)
2. Analisi — muscle balance radar + bar list (bars switch to "% of reference athlete"
   when the reference checkbox is on) → weekly RTV trend per muscle (stacked bars,
   12/26/52 weeks, reference total as dotted line) → Costanza (26-week session
   calendar + stat tiles) → Progressione (records/stall table, load + e1RM chart,
   RTV chart) → AI coaching
3. Profilo — bodyweight log + chart, persistent notes, goal, exercise management tools,
   Hevy import

Chart colours come from the `_C_SERIES` palette in app.py (fixed order, one slot per
muscle in MUSCLES order; light/dark variants picked via `st.context.theme`).

## Key logic

### RTV (Relative Training Volume)
The core metric. Implemented in metrics.py. Formula by exercise type:
- weighted: `(load / (coeff × bodyweight)) × (reps / 10) × sets` — coeff is the
  optional per-exercise `reference_bw_coefficient` (default 1.0)
- bodyweight: `1.0 × (reps / 10) × sets`
- weighted_bw: `((bodyweight + added_weight) / bodyweight) × (reps / 10) × sets`
- timed: `duration / 120` (120s reference, sets not applied)
- excluded: always 0.0

For AMRAP set type: `sets` includes the AMRAP set; formula is `(sets-1)` standard
sets + 1 set at `reps_actual`. The final AMRAP set is within the `sets` count, not
added on top.
For drop_inverse set type: `sets` includes the up-set; formula is `(sets-1)` base
sets + 1 set at `value2 × (reps_actual or reps)`. The drop set (value_drop/reps_drop)
is stored but explicitly excluded from RTV.
For fixed_plus set type: `sets` includes the heavier final set; formula is `(sets-1)`
base sets + 1 set at `value2 × reps`. Falls back to standard when value2 is missing.

Muscle contribution is fractional, not binary. Each exercise has a `muscles`
dict in exercises.json mapping muscle group → weight (summing to 1.0).
RTV is multiplied by these weights when accumulating per-muscle scores.

Period comparison scores are normalized by calendar weeks (RTV per week),
not session count — this makes different period lengths directly comparable
and scales correctly with the reference athlete.

Do not simplify or replace RTV without explicit instruction.

### Exercise library
Defined in data/exercises.json. Loaded at startup by config.py, which merges
with hardcoded fallback defaults. Schema per exercise:
- name, type, day_ids, muscles (dict), set_type, no_amrap, variants, default,
  reference_load, source, exrx_url

`reload_exercises()` in config.py rebuilds DAYS, EX_MUSCLES and the metadata store
in place from the JSON; app.py calls it at the top of every Streamlit run, so
JSON changes apply without restarting. A hardcoded built-in day exercise is
dropped when its JSON record no longer lists that day in `day_ids`.

`get_exercise_meta(name)` in config.py returns the full metadata dict for any
exercise. Used by app.py for UI pre-population and by metrics.py for fractional
RTV per muscle.

`reference_load` is the representative load (kg or seconds) for a 75 kg reference
athlete. Used by `get_reference_rtv_weekly()` to auto-compute the reference athlete
benchmark without any hardcoded per-muscle values. Set to 0 for bodyweight/excluded
exercises and for weighted_bw exercises where 0 added weight is the reference point.

### Reference athlete
Auto-computed by `get_reference_rtv_weekly()` in metrics.py. Uses `reference_load`
at 75 kg bodyweight, 4 sets, 10 reps, and `day_ids` to derive session frequency
(exercises appearing in more days contribute proportionally more). Returns weekly
RTV per muscle in Italian muscle names. There is no hardcoded `REFERENCE_ATHLETE`
in config.py — the JSON is the single source of truth.

### Routine sync on save
When a session is saved, app.py compares the submitted exercises against the
current day's exercise list and writes structural changes back to exercises.json:
- Added exercises: append their `day_ids` entry for this day
- Removed exercises: remove this day from their `day_ids`
- Changed set_type: update `set_type` on the exercise record

Removing a hardcoded-only exercise (no JSON record) creates a JSON record with
the remaining days, so the removal sticks.

Built-in days (D1–D4) use `day_ids` on exercises; custom days have a
`days[i].exercises` list and are handled separately.

### Hevy import
Workouts whose title has no D1–D4 prefix are imported as day 0 "Non assegnato"
(`UNASSIGNED_DAY_ID` in hevy_importer.py): they count in muscle analysis but
never pre-populate a real day's form. The import preview lists them.

### Session data model
sessions.csv columns (one row per exercise per session):
`session_id, date, day_id, day_name, exercise, variant, type, sets, reps,
value, value2, set_type, reps_actual, value_drop, reps_drop, skipped, note`

Old rows missing new columns are backfilled with safe defaults by load_sessions().

Bodyweight per session comes from `make_bodyweight_lookup()` in data_manager.py:
latest measurement on or before the date; sessions before the first measurement
use the earliest one. Always guard bodyweight against NaN (NaN is truthy).
Never add new columns without updating load_sessions() defaults.

### Set types
- standard — straight sets, no special final set
- amrap — final set to technical failure at same weight; logs reps_actual;
  sets count includes the AMRAP set
- drop_inverse — final set at increased weight (value2) to failure, then
  immediate drop to lower weight (value_drop) to failure; logs reps_actual + reps_drop;
  sets count includes the up-set; drop set excluded from RTV
- fixed_plus — final set at increased weight (value2), fixed reps, not to failure;
  sets count includes the final set
- none — no intensity modifier (warmup exercises, timed holds)

### 1RM estimation, records and stalls
`estimate_1rm(load, reps)` in metrics.py uses Epley formula: `load × (1 + reps/30)`.
Never used inside RTV calculations.
`session_e1rm(row)` takes the best e1RM of a logged row: base load × reps, AMRAP
final set × reps_actual, drop_inverse up-set, fixed_plus final set (drop part
ignored). weighted_bw uses total load (bodyweight + added).
`exercise_records(df)` returns per-exercise max load, best e1RM, PR date and
`since_pr` (sessions since the last new best); `>= STALL_SESSIONS` (3) = stalled.

### Other analysis helpers (metrics.py)
- `row_rtv(row)` / `muscle_split(name, rtv)` — single source for per-row RTV and its
  fractional split across muscles; reuse them instead of re-deriving RTV.
- `weekly_muscle_rtv(df, n_weeks)` — Monday-based weekly RTV per muscle.
- `daily_session_counts(df)` — sessions per date for the calendar.

### LLM backend
Configurable via `LLM_BACKEND` flag in llm.py.
- "claude" — Anthropic API, claude-sonnet-4-20250514, uses system parameter
- "ollama" — local Ollama via the `ollama` Python client, system message in messages array
Both backends receive the same system prompt and user message structure.
The prompt always ends with a Dodgeball quote. This is load-bearing for morale.

### LLM session context format
`_format_exercise_line()` in llm.py renders each exercise row for the LLM prompt.
For set types that have a special final set, `base_sets = sets - 1` is used for
the standard portion display, then the final set is appended:
- amrap: `{load} kg × {base_sets}×{reps} + AMRAP {reps_actual} reps`
- drop_inverse: `{load} kg × {base_sets}×{reps} + {value2} kg × {up_reps} reps → drop {value_drop} kg × {reps_drop} reps`
- fixed_plus: `{load} kg × {base_sets}×{reps} + {value2} kg (final)`
- unknown set types: appended as `[set_type]`

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
