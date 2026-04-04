import anthropic
from data_manager import load_sessions, get_bodyweight_on
from metrics import compute_rtv
from config import MUSCLES, EX_MUSCLES


def build_session_summary(n_sessions: int = 10) -> str:
    """
    Costruisce un testo riassuntivo delle ultime N sessioni,
    da passare come contesto al LLM.
    """
    df = load_sessions()
    if df.empty:
        return "Nessuna sessione registrata ancora."

    # prende le ultime N sessioni uniche
    recent_ids = (
        df[['session_id', 'date']]
        .drop_duplicates()
        .sort_values('date', ascending=False)
        .head(n_sessions)['session_id']
        .tolist()
    )
    df_recent = df[df['session_id'].isin(recent_ids)]

    lines = []
    for sid in recent_ids:
        sess = df_recent[df_recent['session_id'] == sid]
        date = sess.iloc[0]['date']
        day_name = sess.iloc[0]['day_name']
        note = sess.iloc[0]['note'] if pd.notna(sess.iloc[0]['note']) else ''
        bw = get_bodyweight_on(date)

        lines.append(f"\n--- {date} | {day_name} ---")
        if bw:
            lines.append(f"Peso corporeo: {bw} kg")
        if note:
            lines.append(f"Note: {note}")

        for _, row in sess.iterrows():
            if row['skipped']:
                lines.append(f"  - {row['exercise']}: SALTATO")
            elif row['type'] == 'excluded':
                continue
            elif row['type'] == 'timed':
                rtv = compute_rtv(row['type'], row['value'], bw or 0)
                lines.append(f"  - {row['exercise']}: {int(row['value'])}s (RTV: {rtv:.2f})")
            else:
                rtv = compute_rtv(row['type'], row['value'], bw or 0)
                lines.append(f"  - {row['exercise']}: {row['value']} kg (RTV: {rtv:.2f})")

    return '\n'.join(lines)


def build_muscle_summary(df_current, df_previous=None) -> str:
    """
    Costruisce un testo con i punteggi muscolari correnti
    e il confronto con il periodo precedente se disponibile.
    """
    from metrics import compute_muscle_scores, normalize_scores

    scores_cur = normalize_scores(compute_muscle_scores(df_current, 'rtv'))

    lines = ["\nDistribuzione carico muscolare (normalizzata 0-1):"]
    for m in MUSCLES:
        bar = '█' * int(scores_cur.get(m, 0) * 10)
        lines.append(f"  {m:<15} {scores_cur.get(m, 0):.2f}  {bar}")

    if df_previous is not None and not df_previous.empty:
        scores_prev = normalize_scores(compute_muscle_scores(df_previous, 'rtv'))
        lines.append("\nVariazione rispetto al periodo precedente:")
        for m in MUSCLES:
            delta = scores_cur.get(m, 0) - scores_prev.get(m, 0)
            arrow = '▲' if delta > 0.05 else ('▼' if delta < -0.05 else '=')
            lines.append(f"  {m:<15} {arrow} {delta:+.2f}")

    return '\n'.join(lines)


def get_llm_analysis(user_profile: dict, df_current, df_previous=None,
                     focus: str = 'general') -> str:
    """
    Manda il contesto dell'allenamento a Claude e restituisce l'analisi.

    focus: 'general' | 'balance' | 'progression' | 'next_session'
    """
    session_summary = build_session_summary()
    muscle_summary = build_muscle_summary(df_current, df_previous)

    focus_instructions = {
        'general': "Dai un'analisi generale dell'andamento degli allenamenti.",
        'balance': "Concentrati sull'equilibrio muscolare. Ci sono gruppi sovra o sotto-allenati?",
        'progression': "Analizza la progressione dei carichi nel tempo. Si sta progredendo?",
        'next_session': "Suggerisci come impostare la prossima sessione basandoti sui dati.",
    }

    prompt = f"""Sei un coach di allenamento esperto. Analizza i dati di allenamento seguenti 
e fornisci feedback concreto e actionable in italiano.

PROFILO ATLETA:
- Peso corporeo attuale: {user_profile.get('bodyweight', 'non disponibile')} kg
- Obiettivo: {user_profile.get('goal', 'non specificato')}
- Note persistenti (infortuni, limitazioni, obiettivi): {user_profile.get('memory', 'nessuna')}

STORICO SESSIONI RECENTI:
{session_summary}

{muscle_summary}

FOCUS DELL'ANALISI:
{focus_instructions.get(focus, focus_instructions['general'])}

Sii diretto, specifico, e basa i tuoi commenti sui numeri reali. 
Evita consigli generici. Massimo 300 parole."""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


import pandas as pd