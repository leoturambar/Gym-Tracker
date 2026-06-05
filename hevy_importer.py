"""Hevy CSV import utilities for gym-tracker."""
import json
import os
from collections import Counter
from datetime import datetime

import pandas as pd

from config import DAYS, get_exercise_meta
from data_manager import SESSIONS_COLS

# ── Constants ─────────────────────────────────────────────────────────────────

_HEVY_MAP_FILE = 'data/hevy_exercise_map.json'

_ITALIAN_MONTHS = {
    'gen': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'mag': 5, 'giu': 6,
    'lug': 7, 'ago': 8, 'set': 9, 'ott': 10, 'nov': 11, 'dic': 12,
}

_DAY_ID_TO_INT = {'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}

_DAY_NAME_MAP = {
    'D1': next((d['name'] for d in DAYS if d['id'] == 1), 'Upper Push'),
    'D2': next((d['name'] for d in DAYS if d['id'] == 2), 'Lower Quad'),
    'D3': next((d['name'] for d in DAYS if d['id'] == 3), 'Upper Pull'),
    'D4': next((d['name'] for d in DAYS if d['id'] == 4), 'Lower Hip'),
}

# Hevy exercise names where assistance weight is stored as a positive number
_ASSISTED_HEVY_NAMES = {'Pull Up (Assisted)'}

# Leg Press resolves to different app exercises depending on the training day
_LEG_PRESS_DAY_MAP = {
    'D2': 'Leg press piedi medi',
    'D4': 'Leg press piedi alti e larghi',
}

DEFAULT_EXERCISE_MAP = {
    "Ab Wheel": "Abs wheel",
    "Back Extension (Weighted Hyperextension)": "Iperestensioni lombari",
    "Band Pullaparts": "Band pull apart",
    "Bicep Curl (Cable)": "Curl bicipiti cavo",
    "Butterfly (Pec Deck)": "Pec fly",
    "Cable Core Pallof Press": "Pallof press",
    "Chest Press (Machine)": "Chest press",
    "Crunch (Machine)": "Abs machine",
    "Face Pull": "Face pull cavo",
    "Glute Kickback (Machine)": "Glute kickback",
    "Hanging Knee Raise": "Hanging knee raise",
    "Hanging Leg Raise": "Hanging leg raise",
    "Hip Abduction (Machine)": "Abduttori macchina",
    "Hip Adduction (Machine)": "Adduttori macchina",
    "Hip Thrust (Barbell)": "Hip thrust",
    "Hollow Rock": "Hollow body hold",
    "Incline Bench Press (Dumbbell)": "Incline chest press manubri",
    "Lat Pulldown (Machine)": "Lat machine presa larga",
    "Lateral Raise (Machine)": "Delts machine",
    "Leg Extension (Machine)": "Leg extension",
    "Leg Press (Machine)": "__DAY_DEPENDENT__",
    "Plank": "Plank",
    "Pull Up (Assisted)": "Trazioni assistite",
    "Romanian Deadlift (Dumbbell)": "Romanian deadlift",
    "Russian Twist (Weighted)": "Russian twists",
    "Scapular Pull Ups": "Scapular pull up",
    "Seated Cable Row - Bar Grip": "Pulley basso",
    "Seated Calf Raise": "Calf raise",
    "Seated Incline Curl (Dumbbell)": "Curl bicipiti panca inclinata 60° manubri",
    "Seated Leg Curl (Machine)": "Leg curl",
    "Seated Row (Machine)": "Rematore",
    "Shoulder Press (Machine Plates)": "Shoulder press",
    "Side Bend": "Addominali laterali panca romana",
    "Side Plank": "Plank laterale",
    "Triceps Dip": "Dips",
    "Triceps Rope Pushdown": "Tricipiti ai cavi",
}

# ── Exercise map I/O ──────────────────────────────────────────────────────────

def load_exercise_map() -> dict:
    """Load exercise name mapping from data/ or return DEFAULT, saving it first."""
    if os.path.exists(_HEVY_MAP_FILE):
        with open(_HEVY_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    save_exercise_map(DEFAULT_EXERCISE_MAP)
    return dict(DEFAULT_EXERCISE_MAP)


def save_exercise_map(mapping: dict):
    """Save exercise name mapping to data/hevy_exercise_map.json."""
    with open(_HEVY_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_hevy_date(s: str) -> datetime:
    """Parse Italian-locale Hevy timestamp: '4 giu 2026, 15:43' → datetime."""
    s = s.strip().replace(',', '')
    parts = s.split()
    # parts: ['4', 'giu', '2026', '15:43'] or ['4', 'giu', '2026', '15:43:00']
    day = int(parts[0])
    month = _ITALIAN_MONTHS[parts[1].lower()]
    year = int(parts[2])
    time_parts = parts[3].split(':')
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    second = int(time_parts[2]) if len(time_parts) > 2 else 0
    return datetime(year, month, day, hour, minute, second)


# ── CSV parsing ───────────────────────────────────────────────────────────────

# Maps internal column names to possible Hevy CSV header variants (lowercase)
_HEVY_COL_CANDIDATES = {
    'title':         ['title'],
    'start_time':    ['start time'],
    'end_time':      ['end time'],
    'exercise_title': ['exercise_title', 'exercise name'],
    'set_order':     ['set order'],
    'weight_kg':     ['weight (kg)', 'weight_kg', 'weight'],
    'reps':          ['reps'],
    'distance_m':    ['distance (m)', 'distance_m'],
    'duration_s':    ['duration (s)', 'duration_s', 'duration'],
    'notes':         ['notes', 'notes (workout)', 'workout notes'],
    'rpe':           ['rpe'],
    'set_type':      ['set type', 'set_type'],
}


def parse_hevy_csv(file_obj) -> pd.DataFrame:
    """Parse a Hevy CSV export. Returns a normalised DataFrame with a _dt column."""
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]

    # Build case-insensitive lookup: lowercase header → original column name
    col_lower = {c.lower(): c for c in df.columns}

    rename = {}
    for internal, candidates in _HEVY_COL_CANDIDATES.items():
        for cand in candidates:
            if cand in col_lower:
                orig = col_lower[cand]
                if orig != internal:
                    rename[orig] = internal
                break

    df = df.rename(columns=rename)

    required = ['title', 'start_time', 'exercise_title']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV mancante di colonne obbligatorie: {', '.join(missing)}")

    df['_dt'] = df['start_time'].apply(_parse_hevy_date)
    return df


# ── Session grouping ──────────────────────────────────────────────────────────

def _extract_day_id(title: str) -> str | None:
    """Return 'D1'–'D4' if the workout title starts with that prefix, else None."""
    for did in ('D1', 'D2', 'D3', 'D4'):
        if str(title).startswith(did):
            return did
    return None


def _resolve_app_name(hevy_name: str, exercise_map: dict, day_id_str: str) -> str | None:
    """Resolve a Hevy exercise name to the app's Italian name, or None if unmapped."""
    mapped = exercise_map.get(hevy_name)
    if mapped is None:
        return None
    if mapped == '__DAY_DEPENDENT__':
        return _LEG_PRESS_DAY_MAP.get(day_id_str)
    return mapped


def group_into_sessions(df: pd.DataFrame) -> list[dict]:
    """
    Group a parsed Hevy DataFrame into session dicts.

    Each session dict:
      {session_id, date, datetime, day_id, title, notes, exercises}
    Each exercise entry (one per Hevy set row):
      {hevy_name, app_name, set_index, set_type, weight, reps, duration_seconds, is_warmup}
    """
    exercise_map = load_exercise_map()
    sessions = []

    for (title, start_time), grp in df.groupby(['title', 'start_time'], sort=False):
        dt = grp['_dt'].iloc[0]
        day_id_str = _extract_day_id(str(title)) or 'D1'
        session_id = int(dt.timestamp())
        date_str = dt.strftime('%Y-%m-%d')

        # First non-null note value for the workout
        notes = ''
        if 'notes' in grp.columns:
            for v in grp['notes'].tolist():
                if pd.notna(v) and str(v).strip() not in ('', 'nan'):
                    notes = str(v).strip()
                    break

        exercises = []
        for _, row in grp.iterrows():
            hevy_name = str(row.get('exercise_title', '')).strip()
            hevy_set_type = str(row.get('set_type', 'normal')).strip().lower()
            is_warmup = hevy_set_type == 'warmup'

            # Weight — negated for assisted pull-up exercises
            raw_w = row.get('weight_kg')
            try:
                weight = float(raw_w) if pd.notna(raw_w) else 0.0
            except (ValueError, TypeError):
                weight = 0.0
            if hevy_name in _ASSISTED_HEVY_NAMES:
                weight = -abs(weight)

            # Duration in seconds (only populated for timed exercises in Hevy)
            raw_dur = row.get('duration_s')
            try:
                dur_val = float(raw_dur) if pd.notna(raw_dur) else 0.0
            except (ValueError, TypeError):
                dur_val = 0.0
            duration_seconds = dur_val if dur_val > 0 else None

            # Reps
            raw_reps = row.get('reps')
            try:
                reps = int(float(raw_reps)) if pd.notna(raw_reps) else 0
            except (ValueError, TypeError):
                reps = 0

            # Set order index
            raw_order = row.get('set_order', 0)
            try:
                set_index = int(float(raw_order)) if pd.notna(raw_order) else 0
            except (ValueError, TypeError):
                set_index = 0

            app_name = _resolve_app_name(hevy_name, exercise_map, day_id_str)

            exercises.append({
                'hevy_name':        hevy_name,
                'app_name':         app_name,
                'set_index':        set_index,
                'set_type':         hevy_set_type,
                'weight':           weight,
                'reps':             reps,
                'duration_seconds': duration_seconds,
                'is_warmup':        is_warmup,
            })

        sessions.append({
            'session_id': session_id,
            'date':       date_str,
            'datetime':   dt,
            'day_id':     day_id_str,
            'title':      str(title),
            'notes':      notes,
            'exercises':  exercises,
        })

    return sessions


# ── Overlap detection ─────────────────────────────────────────────────────────

def find_overlap(sessions: list, existing_dates: set) -> tuple[list, list]:
    """
    Split sessions into (existing, new) based on whether their date already
    appears in existing_dates (set of 'YYYY-MM-DD' strings).
    """
    existing = [s for s in sessions if s['date'] in existing_dates]
    new = [s for s in sessions if s['date'] not in existing_dates]
    return existing, new


# ── Exercise aggregation helpers ──────────────────────────────────────────────

def _mode(values: list):
    """Return the most common value in a list; last element on tie."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _aggregate_session_exercises(session: dict) -> dict[str, dict]:
    """
    Aggregate per-set exercise entries in a session dict into one summary
    dict per app exercise name, skipping warmup sets and unmapped exercises.

    Returns {app_name: {sets, reps, value, set_type, reps_actual,
                        value2, value_drop, reps_drop}}.
    """
    # Group sets by app_name (preserving insertion order = set order)
    by_exercise: dict[str, list[dict]] = {}
    for s in session['exercises']:
        if s['is_warmup'] or s['app_name'] is None:
            continue
        by_exercise.setdefault(s['app_name'], []).append(s)

    result = {}
    for app_name, sets in by_exercise.items():
        if not sets:
            continue

        # Timed exercises: duration_seconds is populated
        if any(s['duration_seconds'] is not None for s in sets):
            durations = [s['duration_seconds'] for s in sets if s['duration_seconds'] is not None]
            value = _mode(durations) or durations[-1]
            result[app_name] = {
                'sets': len(sets), 'reps': 0, 'value': float(value),
                'set_type': 'none', 'reps_actual': None,
                'value2': None, 'value_drop': None, 'reps_drop': None,
            }
            continue

        # Weighted / bodyweight exercises
        # Separate failure (AMRAP) sets from normal/dropset sets
        normal = [s for s in sets if s['set_type'] != 'failure']
        failure = [s for s in sets if s['set_type'] == 'failure']

        base_sets = normal if normal else sets
        weights = [s['weight'] for s in base_sets]
        reps_list = [s['reps'] for s in base_sets if s['reps'] > 0]

        value = float(_mode(weights) if weights else 0.0)
        reps = int(_mode(reps_list) if reps_list else 10)
        n_sets = len(sets)

        if failure:
            set_type = 'amrap'
            reps_actual = failure[-1]['reps']
        else:
            set_type = 'standard'
            reps_actual = None

        result[app_name] = {
            'sets': n_sets, 'reps': reps, 'value': value,
            'set_type': set_type, 'reps_actual': reps_actual,
            'value2': None, 'value_drop': None, 'reps_drop': None,
        }

    return result


# ── Session comparison ────────────────────────────────────────────────────────

def compare_session(hevy_session: dict, app_rows: pd.DataFrame) -> dict:
    """
    Compare an aggregated Hevy session against existing app rows for the same date.

    Returns:
      {date, day_id, only_in_hevy, only_in_app, load_diffs, is_clean}
    where load_diffs is a list of {exercise, app_value, hevy_value}.
    """
    hevy_agg = _aggregate_session_exercises(hevy_session)
    hevy_names = set(hevy_agg.keys())

    app_names: set = set()
    if not app_rows.empty:
        app_names = set(app_rows.loc[~app_rows['skipped'].astype(bool), 'exercise'].tolist())

    only_in_hevy = sorted(hevy_names - app_names)
    only_in_app = sorted(app_names - hevy_names)

    load_diffs = []
    for name in sorted(hevy_names & app_names):
        hevy_val = hevy_agg[name]['value']
        app_val_series = app_rows.loc[app_rows['exercise'] == name, 'value']
        if app_val_series.empty:
            continue
        app_val = float(app_val_series.iloc[0])
        # Flag if difference exceeds 2 kg absolute or 5% relative
        if abs(hevy_val - app_val) > max(2.0, abs(app_val) * 0.05):
            load_diffs.append({
                'exercise':   name,
                'app_value':  app_val,
                'hevy_value': hevy_val,
            })

    is_clean = not (only_in_hevy or only_in_app or load_diffs)

    return {
        'date':         hevy_session['date'],
        'day_id':       hevy_session['day_id'],
        'only_in_hevy': only_in_hevy,
        'only_in_app':  only_in_app,
        'load_diffs':   load_diffs,
        'is_clean':     is_clean,
    }


# ── CSV row construction ──────────────────────────────────────────────────────

def sessions_to_csv_rows(new_sessions: list, exercise_map: dict) -> pd.DataFrame:
    """
    Convert new sessions to a DataFrame matching sessions.csv schema.
    App names are already resolved in each session's exercise entries;
    exercise_map is accepted for API compatibility but not re-applied here.
    Exercises without a resolved app_name are skipped.
    """
    rows = []

    for session in new_sessions:
        day_id_str = session['day_id']
        # Convert "D2" → 2; fall back to stripping prefix for custom days
        stripped = day_id_str.lstrip('D')
        day_int = _DAY_ID_TO_INT.get(day_id_str, int(stripped) if stripped.isdigit() else 1)
        day_name = _DAY_NAME_MAP.get(day_id_str, day_id_str)

        agg = _aggregate_session_exercises(session)

        for app_name, ex in agg.items():
            meta = get_exercise_meta(app_name)
            ex_type = meta.get('type', 'weighted') if meta else 'weighted'

            rows.append({
                'session_id':  session['session_id'],
                'date':        session['date'],
                'day_id':      day_int,
                'day_name':    day_name,
                'exercise':    app_name,
                'variant':     '',
                'type':        ex_type,
                'sets':        int(ex['sets']),
                'reps':        int(ex['reps']),
                'value':       float(ex['value']),
                'value2':      ex['value2'],
                'set_type':    ex['set_type'],
                'reps_actual': ex['reps_actual'],
                'value_drop':  ex['value_drop'],
                'reps_drop':   ex['reps_drop'],
                'skipped':     False,
                'note':        session['notes'],
            })

    if not rows:
        return pd.DataFrame(columns=SESSIONS_COLS)

    return pd.DataFrame(rows)
