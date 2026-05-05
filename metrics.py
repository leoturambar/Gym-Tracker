import pandas as pd
from config import MUSCLES, EX_MUSCLES, TIMED_REFERENCE, get_exercise_meta
from data_manager import load_sessions, get_bodyweight_on

# English JSON muscle keys → Italian MUSCLES names (same mapping as config.py)
_MUSCLE_KEY_MAP = {
    'chest':      'Petto',
    'shoulders':  'Spalle',
    'triceps':    'Tricipiti',
    'back':       'Schiena',
    'biceps':     'Bicipiti',
    'quads':      'Quadricipiti',
    'hamstrings': 'Femorali',
    'core':       'Core',
    'glutes':     'Femorali',    # merged into Femorali (closest existing group)
    'calves':     'Quadricipiti',  # matches existing Calf raise mapping
}


# ── RTV per singolo esercizio ─────────────────────────────────────────────────

def compute_rtv(ex_type: str, value: float, bw: float,
                sets: int = 1, reps: int = 10,
                set_type: str = 'standard',
                value2: float | None = None,
                reps_actual: int | None = None,
                value_drop: float | None = None,
                reps_drop: int | None = None) -> float:
    """
    Compute volume-weighted RTV for one exercise entry.

    Per-set formula by type:
      weighted:    (load / bw) × (reps / 10)
      bodyweight:  1.0 × (reps / 10)
      weighted_bw: ((bw + added) / bw) × (reps / 10)  (added < 0 = assisted)
      timed:       duration / TIMED_REFERENCE  (sets not applied)
      excluded:    0.0

    Full entry: RTV_set × sets, except:
      amrap:        (sets-1) standard sets + 1 AMRAP set at reps_actual
                    (final set is included in sets count, not added on top)
      drop_inverse: (sets-1) base sets + 1 set at value2×(reps_actual or reps)
                    drop set stored but excluded from RTV; falls back to standard if value2 is None

    Default sets=1 preserves backward compatibility for callers that
    pre-date the schema extension and don't supply sets/reps.
    """
    if ex_type == 'excluded':
        return 0.0
    if ex_type == 'timed':
        return value / TIMED_REFERENCE

    def _single_pass(load: float, n_sets: int, n_reps: int) -> float:
        rep_factor = n_reps / 10.0
        if ex_type == 'weighted':
            return (load / bw * rep_factor * n_sets) if bw else 0.0
        elif ex_type == 'bodyweight':
            return 1.0 * rep_factor * n_sets
        elif ex_type == 'weighted_bw':
            return ((bw + load) / bw * rep_factor * n_sets) if bw else 0.0
        return 0.0

    if set_type == 'amrap' and reps_actual is not None:
        # Final set is included within sets count: (sets-1) standard + 1 AMRAP
        return _single_pass(value, sets - 1, reps) + _single_pass(value, 1, reps_actual)

    if set_type == 'drop_inverse' and value2 is not None:
        # (sets-1) base sets + 1 up-set at value2; drop set logged but excluded from RTV per spec
        up_reps = reps_actual if reps_actual is not None else reps
        return _single_pass(value, sets - 1, reps) + _single_pass(value2, 1, up_reps)

    return _single_pass(value, sets, reps)


# ── Muscle scores da una lista di sessioni ────────────────────────────────────

def compute_muscle_scores(df: pd.DataFrame, metric: str = 'freq',
                          period_weeks: float = 1.0) -> dict:
    """
    Compute per-muscle scores from a session DataFrame, normalised by weeks.

    metric:
        'freq' → exercises per week targeting each muscle
        'rtv'  → volume-weighted RTV per week per muscle, scaled by fractional
                 contribution weights from exercises.json via get_exercise_meta().
                 Falls back to EX_MUSCLES equal-weight split when meta is absent.

    period_weeks: calendar weeks covered by df (e.g. 1 for a 7-day slice,
        days_in_month/7 for a month). Returned values are **RTV per week**
        regardless of period length, making all periods directly comparable.
    Requires a 'bodyweight' column in df for the 'rtv' metric.
    """
    scores = {m: 0.0 for m in MUSCLES}

    if df.empty:
        return scores

    for _, row in df[~df['skipped']].iterrows():
        ex_name = row['exercise']

        if metric == 'freq':
            for m in EX_MUSCLES.get(ex_name, []):
                scores[m] += 1.0
            continue

        # rtv path
        if row['type'] == 'excluded':
            continue

        bw = row.get('bodyweight', 0) or 0

        def _get(col, default):
            v = row.get(col, default)
            return default if (v is None or (isinstance(v, float) and pd.isna(v))) else v

        sets        = int(_get('sets', 1))
        reps        = int(_get('reps', 10))
        set_type    = str(_get('set_type', 'standard'))
        v2_raw      = _get('value2', None)
        value2      = float(v2_raw) if v2_raw is not None else None
        ra_raw      = _get('reps_actual', None)
        reps_actual = int(ra_raw) if ra_raw is not None else None
        vd_raw      = _get('value_drop', None)
        value_drop  = float(vd_raw) if vd_raw is not None else None
        rd_raw      = _get('reps_drop', None)
        reps_drop   = int(rd_raw) if rd_raw is not None else None

        rtv = compute_rtv(
            row['type'], row['value'], bw,
            sets=sets, reps=reps, set_type=set_type,
            value2=value2, reps_actual=reps_actual,
            value_drop=value_drop, reps_drop=reps_drop,
        )

        # Fractional muscle weights from exercises.json
        meta = get_exercise_meta(ex_name)
        muscles_raw = meta.get('muscles', {})

        if muscles_raw:
            for eng_key, weight in muscles_raw.items():
                italian = _MUSCLE_KEY_MAP.get(eng_key)
                if italian and italian in MUSCLES:
                    scores[italian] += rtv * weight
        else:
            # Fall back: EX_MUSCLES binary, equal weight per muscle
            binary = [m for m in EX_MUSCLES.get(ex_name, []) if m in MUSCLES]
            if binary:
                equal_w = 1.0 / len(binary)
                for m in binary:
                    scores[m] += rtv * equal_w

    n = max(period_weeks, 1 / 7)
    return {m: v / n for m, v in scores.items()}


def get_reference_rtv_weekly(sessions_per_week: int = 4) -> dict:
    """
    Compute weekly reference-athlete RTV per muscle from exercises.json.

    Uses reference_load (kg or seconds) at 75 kg bodyweight, 4 sets, 10 reps.
    Each exercise is weighted by how many days it appears in relative to the
    total unique training days, scaled by sessions_per_week.
    Returns {italian_muscle_name: weekly_rtv}.
    """
    import json as _json
    import os as _os

    scores = {m: 0.0 for m in MUSCLES}

    path = 'data/exercises.json'
    if not _os.path.exists(path):
        return scores

    with open(path, 'r', encoding='utf-8') as _f:
        _data = _json.load(_f)
    exercises = _data.get('exercises', _data) if isinstance(_data, dict) else _data

    # Count unique valid training days across all exercises
    all_day_ids: set = set()
    for ex in exercises:
        for did in ex.get('day_ids', []):
            if did and did.strip():
                all_day_ids.add(did)
    total_days = len(all_day_ids) or 1

    REF_BW   = 75.0
    REF_SETS = 4
    REF_REPS = 10

    for ex in exercises:
        ex_type = ex.get('type', 'weighted')
        if ex_type == 'excluded':
            continue

        muscles_dict = ex.get('muscles', {})
        if not muscles_dict:
            continue

        valid_days = [d for d in ex.get('day_ids', []) if d and d.strip()]
        if not valid_days:
            continue

        ref_load = float(ex.get('reference_load', 0) or 0)
        frequency = len(valid_days) * sessions_per_week / total_days

        if ex_type == 'timed':
            rtv = ref_load / TIMED_REFERENCE
        elif ex_type == 'bodyweight':
            rtv = 1.0 * (REF_REPS / 10.0) * REF_SETS
        elif ex_type == 'weighted':
            rtv = (ref_load / REF_BW) * (REF_REPS / 10.0) * REF_SETS if REF_BW else 0.0
        elif ex_type == 'weighted_bw':
            rtv = ((REF_BW + ref_load) / REF_BW) * (REF_REPS / 10.0) * REF_SETS if REF_BW else 0.0
        else:
            continue

        weekly_rtv = rtv * frequency
        for eng_key, weight in muscles_dict.items():
            italian = _MUSCLE_KEY_MAP.get(eng_key)
            if italian and italian in MUSCLES:
                scores[italian] += weekly_rtv * weight

    return scores


def normalize_scores(scores: dict) -> dict:
    """Normalizza i punteggi a 0-1 rispetto al massimo."""
    max_val = max(scores.values()) if scores else 1
    if max_val == 0:
        return {m: 0.0 for m in scores}
    return {m: v / max_val for m, v in scores.items()}


def estimate_1rm(load: float, reps: int) -> float:
    """Estimate 1RM using Epley formula: load × (1 + reps/30).
    Returns 0.0 if inputs are invalid."""
    if load <= 0 or reps <= 0:
        return 0.0
    return load * (1 + reps / 30.0)


# ── Aggiunge colonna bodyweight al DataFrame ──────────────────────────────────

def enrich_with_bodyweight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge una colonna 'bodyweight' al DataFrame delle sessioni,
    usando il peso corporeo valido per la data di ogni sessione.
    Questo garantisce che il passato non venga mai ricalcolato con il peso attuale.
    """
    df = df.copy()
    df['bodyweight'] = df['date'].apply(get_bodyweight_on)
    return df


# ── Filtraggio per periodo ────────────────────────────────────────────────────

def filter_by_period(df: pd.DataFrame, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide il DataFrame in due periodi: corrente e precedente.

    period: 'week' | 'month' | 'year' | 'all'

    Restituisce (df_current, df_previous).
    """
    today = pd.Timestamp.today().normalize()

    if period == 'week':
        start_cur = today - pd.Timedelta(days=7)
        start_prev = today - pd.Timedelta(days=14)
        end_prev = start_cur
    elif period == 'month':
        start_cur = today.replace(day=1)
        start_prev = (start_cur - pd.Timedelta(days=1)).replace(day=1)
        end_prev = start_cur
    elif period == 'year':
        start_cur = today.replace(month=1, day=1)
        start_prev = start_cur.replace(year=start_cur.year - 1)
        end_prev = start_cur
    else:  # all
        return df, pd.DataFrame(columns=df.columns)

    dates = pd.to_datetime(df['date'])
    df_cur = df[dates >= start_cur]
    df_prev = df[(dates >= start_prev) & (dates < end_prev)]

    return df_cur, df_prev


# ── Progressione di un esercizio nel tempo ────────────────────────────────────

def exercise_progression(exercise: str) -> pd.DataFrame:
    """
    Restituisce un DataFrame con la progressione di un esercizio nel tempo.
    Colonne: date, value, rtv
    """
    df = load_sessions()
    if df.empty:
        return pd.DataFrame(columns=['date', 'value', 'rtv'])

    df = enrich_with_bodyweight(df)
    df_ex = df[(df['exercise'] == exercise) & (~df['skipped'])].copy()

    if df_ex.empty:
        return pd.DataFrame(columns=['date', 'value', 'rtv'])

    def _row_rtv(r):
        def _fget(col, default):
            v = r.get(col, default)
            return default if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        v2 = _fget('value2', None)
        ra = _fget('reps_actual', None)
        vd = _fget('value_drop', None)
        rd = _fget('reps_drop', None)
        return compute_rtv(
            r['type'], r['value'], r['bodyweight'] or 0,
            sets=int(_fget('sets', 1)), reps=int(_fget('reps', 10)),
            set_type=str(_fget('set_type', 'standard')),
            value2=float(v2) if v2 is not None else None,
            reps_actual=int(ra) if ra is not None else None,
            value_drop=float(vd) if vd is not None else None,
            reps_drop=int(rd) if rd is not None else None,
        )
    df_ex['rtv'] = df_ex.apply(_row_rtv, axis=1)

    return df_ex[['date', 'value', 'rtv']].sort_values('date').reset_index(drop=True)