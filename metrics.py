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
                reps_actual: int | None = None) -> float:
    """
    Compute volume-weighted RTV for one exercise entry.

    Per-set formula by type:
      weighted:    (load / bw) × (reps / 10)
      bodyweight:  1.0 × (reps / 10)
      weighted_bw: ((bw + added) / bw) × (reps / 10)  (added < 0 = assisted)
      timed:       duration / TIMED_REFERENCE  (sets not applied)
      excluded:    0.0

    Full entry: RTV_set × sets, except:
      amrap:        uses reps_actual in place of reps when available
      drop_inverse: base_load×sets×reps + value2×1×(reps_actual or reps)
                    falls back to standard if value2 is None

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

    effective_reps = reps_actual if (set_type == 'amrap' and reps_actual is not None) else reps

    if set_type == 'drop_inverse' and value2 is not None:
        drop_reps = reps_actual if reps_actual is not None else reps
        return _single_pass(value, sets, effective_reps) + _single_pass(value2, 1, drop_reps)

    return _single_pass(value, sets, effective_reps)


# ── Muscle scores da una lista di sessioni ────────────────────────────────────

def compute_muscle_scores(df: pd.DataFrame, metric: str = 'freq') -> dict:
    """
    Compute per-muscle scores from a session DataFrame, normalised by session count.

    metric:
        'freq' → average number of times each muscle is targeted per session
        'rtv'  → average volume-weighted RTV per session per muscle, scaled by
                 fractional contribution weights from exercises.json via
                 get_exercise_meta(). Falls back to EX_MUSCLES equal-weight
                 split when meta is absent.

    Returned values are **RTV per session** (or exercises per session for freq),
    NOT cumulative totals. Dividing by session count makes scores from periods
    of different lengths directly comparable on the radar chart.
    Requires a 'bodyweight' column in df for the 'rtv' metric.
    """
    scores = {m: 0.0 for m in MUSCLES}

    if df.empty:
        return scores

    session_count = df['session_id'].nunique() if 'session_id' in df.columns else 1

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

        rtv = compute_rtv(
            row['type'], row['value'], bw,
            sets=sets, reps=reps, set_type=set_type,
            value2=value2, reps_actual=reps_actual,
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

    n = max(session_count, 1)
    return {m: v / n for m, v in scores.items()}


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

    df_ex['rtv'] = df_ex.apply(
        lambda r: compute_rtv(r['type'], r['value'], r['bodyweight'] or 0), axis=1
    )

    return df_ex[['date', 'value', 'rtv']].sort_values('date').reset_index(drop=True)