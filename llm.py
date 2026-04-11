import pandas as pd
from data_manager import get_bodyweight_on
from metrics import compute_rtv
from config import MUSCLES, EX_MUSCLES

# ── Selezione backend LLM ─────────────────────────────────────────────────────
# Cambia questo flag per switchare tra Claude e Ollama
LLM_BACKEND = "ollama"  # "claude" oppure "ollama"

OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

CLAUDE_MODEL = "claude-sonnet-4-20250514"


def get_client():
    """Restituisce il client LLM appropriato in base al backend scelto."""
    if LLM_BACKEND == "claude":
        import anthropic
        return anthropic.Anthropic()
    else:
        from openai import OpenAI
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # valore richiesto ma non verificato da Ollama
        )


def call_llm(prompt: str) -> str:
    """Chiama il backend LLM selezionato e restituisce la risposta."""
    if LLM_BACKEND == "claude":
        client = get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    else:
        client = get_client()
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# ── Costruzione contesto ──────────────────────────────────────────────────────

def build_session_summary(df: pd.DataFrame, n_sessions: int = 10) -> str:
    """
    Costruisce un testo riassuntivo delle ultime N sessioni nel DataFrame fornito,
    da passare come contesto al LLM.
    df deve avere già la colonna 'bodyweight' (via enrich_with_bodyweight).
    """
    if df.empty:
        return "Nessuna sessione registrata ancora."

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
        bw = sess.iloc[0].get('bodyweight') if 'bodyweight' in sess.columns else get_bodyweight_on(date)

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
    """Costruisce testo con distribuzione muscolare e confronto periodo."""
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


# ── Analisi principale ────────────────────────────────────────────────────────

def get_llm_analysis(user_profile: dict, df_current, df_previous=None,
                     focus: str = 'general') -> str:
    """
    Genera analisi dell'allenamento via LLM.
    focus: 'general' | 'balance' | 'progression' | 'next_session'
    """
    session_summary = build_session_summary(df_current)
    muscle_summary = build_muscle_summary(df_current, df_previous)

    focus_instructions = {
        'general':      "Provide a general analysis of training progress and patterns.",
        'balance':      "Focus on muscular balance. Are any groups over or undertrained?",
        'progression':  "Analyze load progression over time. Is there consistent progress?",
        'next_session': "Suggest how to approach the next session based on the data.",
    }

    prompt = f"""You are an expert strength and conditioning coach. Analyze the following training data
and provide concrete, actionable feedback based on the actual numbers.

ATHLETE PROFILE:
- Current body weight: {user_profile.get('bodyweight', 'not available')} kg
- Goal: {user_profile.get('goal', 'not specified')}
- Persistent notes (injuries, limitations, long-term goals): {user_profile.get('memory', 'none')}

RECENT SESSION HISTORY:
{session_summary}

{muscle_summary}

ANALYSIS FOCUS:
{focus_instructions.get(focus, focus_instructions['general'])}

Be direct and specific. Base your comments on the actual numbers provided.
Avoid generic advice. Maximum 500 words. Always end your response with this exact quote:
"If you can dodge a wrench, you can dodge a ball." """

    return call_llm(prompt)