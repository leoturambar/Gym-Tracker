import pandas as pd
from config import MUSCLES, EX_MUSCLES, TIMED_REFERENCE
from data_manager import load_sessions, get_bodyweight_on


# ── RTV per singolo esercizio ─────────────────────────────────────────────────

def compute_rtv(ex_type: str, value: float, bw: float) -> float:
    """
    Calcola il Relative Training Volume per un singolo esercizio.

    weighted:    RTV = value / bw
    bodyweight:  RTV = 1.0
    weighted_bw: RTV = (bw + value) / bw  (value può essere negativo = assisted)
    timed:       RTV = value / TIMED_REFERENCE
    excluded:    RTV = 0.0
    """
    if ex_type == 'weighted':
        return value / bw if bw else 0.0
    elif ex_type == 'bodyweight':
        return 1.0
    elif ex_type == 'weighted_bw':
        return (bw + value) / bw if bw else 0.0
    elif ex_type == 'timed':
        return value / TIMED_REFERENCE
    else:  # excluded
        return 0.0


# ── Muscle scores da una lista di sessioni ────────────────────────────────────

def compute_muscle_scores(df: pd.DataFrame, metric: str = 'freq') -> dict:
    """
    Calcola il punteggio per ogni gruppo muscolare da un DataFrame di sessioni.

    metric:
        'freq' → conta quante volte ogni muscolo viene stimolato
        'rtv'  → somma RTV per ogni muscolo (richiede colonna 'bodyweight' nel df)

    Restituisce dict {muscolo: valore_raw} NON normalizzato.
    """
    scores = {m: 0.0 for m in MUSCLES}

    for _, row in df[~df['skipped']].iterrows():
        muscles = EX_MUSCLES.get(row['exercise'], [])
        if not muscles:
            continue

        if metric == 'freq':
            contrib = 1.0
        else:  # rtv
            bw = row.get('bodyweight', 0) or 0
            contrib = compute_rtv(row['type'], row['value'], bw)

        for m in muscles:
            scores[m] += contrib

    return scores


def normalize_scores(scores: dict) -> dict:
    """Normalizza i punteggi a 0-1 rispetto al massimo."""
    max_val = max(scores.values()) if scores else 1
    if max_val == 0:
        return {m: 0.0 for m in scores}
    return {m: v / max_val for m, v in scores.items()}


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