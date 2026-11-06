import pandas as pd
from config import MUSCLES, EX_MUSCLES, TIMED_REFERENCE, get_exercise_meta, _MUSCLE_KEY_MAP
from data_manager import load_sessions, make_bodyweight_lookup

# ── RTV per singolo esercizio ─────────────────────────────────────────────────

def compute_rtv(ex_type: str, value: float, bw: float,
                sets: int = 1, reps: int = 10,
                set_type: str = 'standard',
                value2: float | None = None,
                reps_actual: int | None = None,
                value_drop: float | None = None,
                reps_drop: int | None = None,
                exercise_name: str | None = None) -> float:
    """
    Compute volume-weighted RTV for one exercise entry.

    Per-set formula by type:
      weighted:    (load / (coeff * bw)) × (reps / 10)  where coeff = reference_bw_coefficient
                   (coeff defaults to 1.0 when not set; RTV=1.0 means intermediate standard)
      bodyweight:  1.0 × (reps / 10)
      weighted_bw: ((bw + added) / bw) × (reps / 10)  (added < 0 = assisted)
      timed:       duration / TIMED_REFERENCE  (sets not applied)
      excluded:    0.0

    Full entry: RTV_set × sets, except:
      amrap:        (sets-1) standard sets + 1 AMRAP set at reps_actual
                    (final set is included in sets count, not added on top)
      drop_inverse: (sets-1) base sets + 1 set at value2×(reps_actual or reps)
                    drop set stored but excluded from RTV; falls back to standard if value2 is None
      fixed_plus:   (sets-1) base sets + 1 set at value2×reps (final set is heavier,
                    same reps, not to failure); falls back to standard if value2 is None

    Default sets=1 preserves backward compatibility for callers that
    pre-date the schema extension and don't supply sets/reps.
    """
    if ex_type == 'excluded':
        return 0.0
    if ex_type == 'timed':
        return value / TIMED_REFERENCE

    ref_coeff = 1.0
    if ex_type == 'weighted' and exercise_name:
        c = get_exercise_meta(exercise_name).get('reference_bw_coefficient')
        if c:
            ref_coeff = float(c)

    def _single_pass(load: float, n_sets: int, n_reps: int) -> float:
        rep_factor = n_reps / 10.0
        if ex_type == 'weighted':
            return (load / (ref_coeff * bw) * rep_factor * n_sets) if bw else 0.0
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

    if set_type == 'fixed_plus' and value2 is not None:
        # (sets-1) base sets + 1 heavier final set at value2, same target reps
        return _single_pass(value, sets - 1, reps) + _single_pass(value2, 1, reps)

    return _single_pass(value, sets, reps)


# ── Per-row helpers ──────────────────────────────────────────────────────────

def _clean(row, col, default=None):
    """Read a row field, returning default for missing / None / NaN."""
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v


def row_rtv(row) -> float:
    """RTV of one session row. Requires a 'bodyweight' column (enrich_with_bodyweight)."""
    if row['type'] == 'excluded':
        return 0.0
    v2 = _clean(row, 'value2')
    ra = _clean(row, 'reps_actual')
    vd = _clean(row, 'value_drop')
    rd = _clean(row, 'reps_drop')
    return compute_rtv(
        row['type'], float(_clean(row, 'value', 0.0)), _valid_bw(row.get('bodyweight')),
        sets=int(_clean(row, 'sets', 1)), reps=int(_clean(row, 'reps', 10)),
        set_type=str(_clean(row, 'set_type', 'standard')),
        value2=float(v2) if v2 is not None else None,
        reps_actual=int(ra) if ra is not None else None,
        value_drop=float(vd) if vd is not None else None,
        reps_drop=int(rd) if rd is not None else None,
        exercise_name=row['exercise'],
    )


def muscle_split(ex_name: str, rtv: float) -> dict:
    """
    Split an exercise RTV across muscles using the fractional weights in
    exercises.json; falls back to an equal split over EX_MUSCLES.
    """
    out: dict = {}
    muscles_raw = get_exercise_meta(ex_name).get('muscles', {})
    if muscles_raw:
        for eng_key, weight in muscles_raw.items():
            italian = _MUSCLE_KEY_MAP.get(eng_key)
            if italian and italian in MUSCLES:
                out[italian] = out.get(italian, 0.0) + rtv * weight
    else:
        binary = [m for m in EX_MUSCLES.get(ex_name, []) if m in MUSCLES]
        for m in binary:
            out[m] = out.get(m, 0.0) + rtv / len(binary)
    return out


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

    for _, row in df[~df['skipped'].astype(bool)].iterrows():
        ex_name = row['exercise']

        if metric == 'freq':
            for m in EX_MUSCLES.get(ex_name, []):
                scores[m] += 1.0
            continue

        if row['type'] == 'excluded':
            continue
        for m, v in muscle_split(ex_name, row_rtv(row)).items():
            scores[m] += v

    n = max(period_weeks, 1 / 7)
    return {m: v / n for m, v in scores.items()}


def weekly_muscle_rtv(df: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """
    RTV per muscle for each of the last n_weeks calendar weeks (Monday-based,
    current week included even if partial). Weeks without training are 0.
    Returns a DataFrame indexed by week start, one column per muscle in MUSCLES.
    Requires a 'bodyweight' column (enrich_with_bodyweight).
    """
    today = pd.Timestamp.today().normalize()
    this_monday = today - pd.Timedelta(days=today.weekday())
    weeks = pd.DatetimeIndex(
        [this_monday - pd.Timedelta(weeks=i) for i in range(n_weeks - 1, -1, -1)]
    )
    out = pd.DataFrame(0.0, index=weeks, columns=MUSCLES)
    if df.empty:
        return out

    d = df[~df['skipped'].astype(bool) & (df['type'] != 'excluded')]
    dates = pd.to_datetime(d['date'])
    week_start = dates - pd.to_timedelta(dates.dt.weekday, unit='D')
    for (_, row), wk in zip(d.iterrows(), week_start):
        if wk not in out.index:
            continue
        for m, v in muscle_split(row['exercise'], row_rtv(row)).items():
            out.at[wk, m] += v
    return out


def daily_session_counts(df: pd.DataFrame) -> pd.Series:
    """Number of distinct sessions per date ('YYYY-MM-DD' index)."""
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby('date')['session_id'].nunique()


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
            coeff = float(ex.get('reference_bw_coefficient') or 1.0)
            rtv = (ref_load / (coeff * REF_BW)) * (REF_REPS / 10.0) * REF_SETS if REF_BW else 0.0
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
    lookup = make_bodyweight_lookup()  # reads bodyweight.csv once, not per row
    df['bodyweight'] = df['date'].apply(lookup)
    return df


def _valid_bw(v) -> float:
    """Return bodyweight as float, or 0.0 for None/NaN (NaN is truthy, so `or 0` is not enough)."""
    if v is None or pd.isna(v):
        return 0.0
    return float(v)


# ── Filtraggio per periodo ────────────────────────────────────────────────────

def filter_by_period(df: pd.DataFrame, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide il DataFrame in due periodi: corrente e precedente.

    period: 'week' | 'month' | 'year' | 'all'

    Restituisce (df_current, df_previous).
    """
    today = pd.Timestamp.today().normalize()

    if period == 'week':
        # 7 days each, today included: [today-6, today] vs [today-13, today-7]
        start_cur = today - pd.Timedelta(days=6)
        start_prev = today - pd.Timedelta(days=13)
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

def session_e1rm(row) -> float | None:
    """
    Best estimated 1RM (Epley) across the sets logged in one row, in kg.

    weighted:    external load. weighted_bw: total load (bodyweight + added),
    None without a bodyweight. Other types: None.
    Candidates: base load × reps; AMRAP final set × reps_actual;
    drop_inverse up-set value2 × (reps_actual or reps); fixed_plus value2 × reps.
    The drop part of drop_inverse is ignored (fatigued, not a strength signal).
    """
    ex_type = row['type']
    if ex_type not in ('weighted', 'weighted_bw'):
        return None
    offset = 0.0
    if ex_type == 'weighted_bw':
        offset = _valid_bw(row.get('bodyweight'))
        if offset <= 0:
            return None

    value    = float(_clean(row, 'value', 0.0))
    reps     = int(_clean(row, 'reps', 10))
    set_type = str(_clean(row, 'set_type', 'standard'))
    value2   = _clean(row, 'value2')
    ra       = _clean(row, 'reps_actual')

    cands = [estimate_1rm(value + offset, reps)]
    if set_type == 'amrap' and ra is not None:
        cands.append(estimate_1rm(value + offset, int(ra)))
    elif set_type == 'drop_inverse' and value2 is not None:
        cands.append(estimate_1rm(float(value2) + offset,
                                  int(ra) if ra is not None else reps))
    elif set_type == 'fixed_plus' and value2 is not None:
        cands.append(estimate_1rm(float(value2) + offset, reps))
    best = max(cands)
    return best if best > 0 else None


_PROG_COLS = ['date', 'session_id', 'value', 'rtv', 'e1rm',
              'sets', 'reps', 'set_type', 'reps_actual', 'value2']


def exercise_progression(exercise: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Progression of one exercise over time, one row per logged entry.
    Columns: date, session_id, value, rtv, e1rm, sets, reps, set_type, reps_actual, value2.
    Pass an already enriched df (with 'bodyweight') to avoid reloading.
    """
    if df is None:
        df = load_sessions()
        if df.empty:
            return pd.DataFrame(columns=_PROG_COLS)
        df = enrich_with_bodyweight(df)

    df_ex = df[(df['exercise'] == exercise) & (~df['skipped'].astype(bool))].copy()
    if df_ex.empty:
        return pd.DataFrame(columns=_PROG_COLS)

    df_ex['rtv'] = df_ex.apply(row_rtv, axis=1)
    df_ex['e1rm'] = df_ex.apply(session_e1rm, axis=1)
    return df_ex[_PROG_COLS].sort_values(['date', 'session_id']).reset_index(drop=True)


STALL_SESSIONS = 3  # sessions without a new best before an exercise counts as stalled


def exercise_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Personal records and stall detection for weighted / weighted_bw exercises.

    Strength signal per session = best e1RM (falls back to load when e1RM is
    unavailable, e.g. weighted_bw without bodyweight). 'since_pr' counts the
    sessions logged after the last one that set a new best; >= STALL_SESSIONS
    means stalled. Requires a 'bodyweight' column.
    Columns: exercise, type, sessions, max_load, best_e1rm, pr_date, last_date, since_pr.
    """
    cols = ['exercise', 'type', 'sessions', 'max_load', 'best_e1rm',
            'pr_date', 'last_date', 'since_pr']
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df[~df['skipped'].astype(bool)
           & df['type'].isin(['weighted', 'weighted_bw'])].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d['e1rm'] = d.apply(session_e1rm, axis=1)
    d['signal'] = d['e1rm'].where(d['e1rm'].notna(), d['value'].astype(float))

    rows = []
    for ex, g in d.groupby('exercise'):
        per_sess = (g.groupby(['date', 'session_id'])
                     .agg(signal=('signal', 'max'), value=('value', 'max'),
                          e1rm=('e1rm', 'max'))
                     .reset_index()
                     .sort_values(['date', 'session_id']))
        best, pr_idx = None, 0
        for i, sig in enumerate(per_sess['signal']):
            if best is None or sig > best + 1e-9:
                best, pr_idx = sig, i
        has_e1rm = per_sess['e1rm'].notna().any()
        rows.append({
            'exercise':  ex,
            'type':      g['type'].iloc[-1],
            'sessions':  len(per_sess),
            'max_load':  float(per_sess['value'].max()),
            'best_e1rm': float(per_sess['e1rm'].max()) if has_e1rm else None,
            'pr_date':   per_sess['date'].iloc[pr_idx],
            'last_date': per_sess['date'].iloc[-1],
            'since_pr':  len(per_sess) - 1 - pr_idx,
        })
    return pd.DataFrame(rows, columns=cols)
