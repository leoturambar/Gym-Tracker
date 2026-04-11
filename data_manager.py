import pandas as pd
import os
from datetime import date

SESSIONS_FILE = 'data/sessions.csv'
BW_FILE = 'data/bodyweight.csv'

# ── Colonne del CSV sessioni ──────────────────────────────────────────────────
SESSIONS_COLS = [
    'session_id',   # timestamp unix, identificatore unico
    'date',         # YYYY-MM-DD
    'day_id',       # 1-4
    'day_name',     # es. "Upper 1"
    'exercise',     # nome esercizio
    'type',         # weighted / bodyweight / weighted_bw / timed / excluded
    'value',        # kg oppure secondi, a seconda del tipo
    'skipped',      # True/False
    'note',         # nota libera sulla sessione (stessa per tutti gli esercizi della sessione)
]

# ── Colonne del CSV peso corporeo ─────────────────────────────────────────────
BW_COLS = ['date', 'bodyweight']


# ── Inizializzazione file ─────────────────────────────────────────────────────

def init_files():
    """Crea i CSV se non esistono ancora."""
    if not os.path.exists(SESSIONS_FILE):
        pd.DataFrame(columns=SESSIONS_COLS).to_csv(SESSIONS_FILE, index=False)
    if not os.path.exists(BW_FILE):
        pd.DataFrame(columns=BW_COLS).to_csv(BW_FILE, index=False)


# ── Peso corporeo ─────────────────────────────────────────────────────────────

def save_bodyweight(bw: float, on_date: str = None):
    """Salva una nuova misurazione del peso corporeo."""
    if on_date is None:
        on_date = str(date.today())
    df = load_bodyweight()
    # sostituisce se esiste già una misurazione per quella data
    df = df[df['date'] != on_date]
    new_row = pd.DataFrame([{'date': on_date, 'bodyweight': bw}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(BW_FILE, index=False)


def load_bodyweight() -> pd.DataFrame:
    """Carica la storia del peso corporeo."""
    if not os.path.exists(BW_FILE):
        return pd.DataFrame(columns=BW_COLS)
    return pd.read_csv(BW_FILE)


def get_bodyweight_on(on_date: str) -> float | None:
    """Restituisce il peso corporeo valido per una data specifica.
    Usa l'ultima misurazione registrata prima o uguale a quella data."""
    df = load_bodyweight()
    if df.empty:
        return None
    df = df[df['date'] <= on_date].sort_values('date', ascending=False)
    if df.empty:
        return None
    return float(df.iloc[0]['bodyweight'])


# ── Sessioni ──────────────────────────────────────────────────────────────────

def save_session(session_id: int, date_str: str, day_id: int, day_name: str,
                 exercises: list[dict], note: str = ''):
    """
    Salva una sessione di allenamento.
    exercises: lista di dict con chiavi name, type, value, skipped
    """
    df = load_sessions()
    rows = []
    for ex in exercises:
        rows.append({
            'session_id': session_id,
            'date':       date_str,
            'day_id':     day_id,
            'day_name':   day_name,
            'exercise':   ex['name'],
            'type':       ex['type'],
            'value':      ex['value'],
            'skipped':    ex['skipped'],
            'note':       note,
        })
    new_rows = pd.DataFrame(rows)
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(SESSIONS_FILE, index=False)


def load_sessions() -> pd.DataFrame:
    """Carica tutte le sessioni."""
    if not os.path.exists(SESSIONS_FILE):
        return pd.DataFrame(columns=SESSIONS_COLS)
    try:
        df = pd.read_csv(SESSIONS_FILE)
        if df.empty:
            return pd.DataFrame(columns=SESSIONS_COLS)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['skipped'] = df['skipped'].astype(bool)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=SESSIONS_COLS)


def delete_session(session_id: int):
    """Elimina una sessione completa dal CSV."""
    df = load_sessions()
    df = df[df['session_id'] != session_id]
    df.to_csv(SESSIONS_FILE, index=False)


def get_last_values(day_id: int) -> dict:
    """
    Restituisce l'ultimo valore registrato per ogni esercizio di un dato giorno.
    Usato per pre-popolare il form con i valori della sessione precedente.
    """
    df = load_sessions()
    if df.empty:
        return {}
    df_day = df[(df['day_id'] == day_id) & (~df['skipped'])]
    if df_day.empty:
        return {}
    last_session = df_day.sort_values('date', ascending=False).iloc[0]['session_id']
    df_last = df_day[df_day['session_id'] == last_session]
    return dict(zip(df_last['exercise'], df_last['value']))

# ── Memoria ───────────────────────────────────────────────────────────────────

MEMORY_FILE = 'data/memory.txt'

def load_memory() -> str:
    """Carica le note persistenti del profilo atleta."""
    if not os.path.exists(MEMORY_FILE):
        return ''
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        return f.read().strip()

def save_memory(text: str):
    """Salva le note persistenti del profilo atleta."""
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        f.write(text)


GOAL_FILE = 'data/goal.txt'

def load_goal() -> str:
    """Carica l'obiettivo di allenamento salvato."""
    if not os.path.exists(GOAL_FILE):
        return ''
    with open(GOAL_FILE, 'r', encoding='utf-8') as f:
        return f.read().strip()

def save_goal(goal: str):
    """Salva l'obiettivo di allenamento."""
    with open(GOAL_FILE, 'w', encoding='utf-8') as f:
        f.write(goal)