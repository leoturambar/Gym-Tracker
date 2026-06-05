import pandas as pd
from data_manager import get_bodyweight_on
from config import MUSCLES, EX_MUSCLES, get_exercise_meta

# ── LLM backend selection ─────────────────────────────────────────────────────
# Change this flag to switch between Claude and Ollama
LLM_BACKEND = "ollama"  # "claude" or "ollama"

OLLAMA_MODEL = "qwen3.5:9b" #"qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert strength and conditioning coach
analyzing personal training data.

SESSION HISTORY lists exercises in this format:
  EXERCISE: LOAD kg × SETS×REPS (X% of intermediate)

"X% of intermediate" is the working weight as a percentage of the
strengthlevel.com intermediate 1RM standard for that exercise,
adjusted for the athlete's bodyweight.
  - 100% = at intermediate standard
  - >100% = above intermediate
  - <100% = still developing toward intermediate
Use this figure to assess relative strength level per exercise.
Never penalize isolation exercises for low absolute kg — evaluate
them through the % figure. A lateral raise or curl at 80% of
intermediate is a solid result.

MUSCLE GROUP SCORES (0.0–1.0) are normalized within the analysis
period: 1.0 = most-trained group this period. Use these for muscle
balance analysis only. They are relative, not absolute.

The athlete's persistent notes contain injuries and hard constraints —
treat them as absolute limits. Never suggest anything that conflicts.

Be direct and specific. Every observation must cite an actual number
from the data. No generic advice. Max 500 words.
"""


def get_client():
    """Return the appropriate LLM client for the selected backend."""
    if LLM_BACKEND == "claude":
        import anthropic
        return anthropic.Anthropic()
    else:
        from openai import OpenAI
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # required by the OpenAI client but not verified by Ollama
        )


def call_llm(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Call the selected LLM backend and return the response text."""
    if LLM_BACKEND == "claude":
        client = get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    else:
        from ollama import Client
        client = Client(host="http://localhost:11434")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            think=False
        )
        return resp.message.content


# ── Context builders ──────────────────────────────────────────────────────────

def _safe_get(row, col, default=None):
    """Read a DataFrame row field, returning default for NaN / missing."""
    v = row.get(col, default)
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    return v


def _format_exercise_line(row, bw: float) -> str | None:
    """
    Format one exercise row for the session history block.
    Returns None for excluded-type exercises (they are silently skipped).
    """
    name = row['exercise']

    if row['skipped']:
        return f"  - {name}: SKIPPED"
    if row['type'] == 'excluded':
        return None

    sets        = int(_safe_get(row, 'sets',        1))
    reps        = int(_safe_get(row, 'reps',        10))
    set_type    = str(_safe_get(row, 'set_type',    'standard'))
    v2_raw      = _safe_get(row, 'value2',      None)
    value2      = float(v2_raw) if v2_raw is not None else None
    ra_raw      = _safe_get(row, 'reps_actual', None)
    reps_actual = int(ra_raw) if ra_raw is not None else None
    vd_raw      = _safe_get(row, 'value_drop',  None)
    value_drop  = float(vd_raw) if vd_raw is not None else None
    rd_raw      = _safe_get(row, 'reps_drop',   None)
    reps_drop   = int(rd_raw) if rd_raw is not None else None

    if row['type'] == 'timed':
        return f"  - {name}: {int(row['value'])}s"

    load    = row['value']
    ex_type = row['type']

    # For amrap/drop_inverse the special final set is counted within sets,
    # so base_sets = sets - 1 are the plain straight sets.
    base_sets = sets - 1 if set_type in ('amrap', 'drop_inverse') else sets

    if ex_type == 'bodyweight':
        label = f"  - {name}: BW × {base_sets}×{reps}"
    elif ex_type == 'weighted_bw':
        sign = '+' if load >= 0 else ''
        label = f"  - {name}: BW {sign}{load:.0f} kg × {base_sets}×{reps}"
    else:
        label = f"  - {name}: {load} kg × {base_sets}×{reps}"

    # Append final-set detail for each special set type
    if set_type == 'amrap' and reps_actual is not None:
        label += f" + AMRAP {reps_actual} reps"
    elif set_type == 'drop_inverse' and value2 is not None:
        up_reps = reps_actual if reps_actual is not None else reps
        label += f" + {value2:.0f} kg × {up_reps} reps"
        if value_drop is not None:
            dr = reps_drop if reps_drop is not None else reps
            label += f" → drop {value_drop:.0f} kg × {dr} reps"
    elif set_type == 'fixed_plus' and value2 is not None:
        label += f" + {value2:.0f} kg (final)"
    elif set_type not in ('standard', 'none', ''):
        label += f" [{set_type}]"

    if ex_type == 'weighted' and bw:
        meta = get_exercise_meta(name)
        coeff = meta.get('reference_bw_coefficient')
        if coeff:
            perf_pct = int((load / (float(coeff) * bw)) * 100)
            label += f" ({perf_pct}% of intermediate)"

    return label


def build_session_summary(df: pd.DataFrame, n_sessions: int = 10) -> str:
    """
    Build a text summary of the most recent N sessions for LLM context.
    Requires a 'bodyweight' column in df (via enrich_with_bodyweight).
    Includes sets, reps, set_type, and reps_actual where available.
    """
    if df.empty:
        return "No sessions recorded yet."

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
        bw = (sess.iloc[0].get('bodyweight')
              if 'bodyweight' in sess.columns
              else get_bodyweight_on(date))

        lines.append(f"\n--- {date} | {day_name} ---")
        if bw:
            lines.append(f"Bodyweight: {bw} kg")
        if note:
            lines.append(f"Note: {note}")

        for _, row in sess.iterrows():
            line = _format_exercise_line(row, bw or 0)
            if line is not None:
                lines.append(line)

    return '\n'.join(lines)


def build_muscle_summary(df_current, df_previous=None) -> str:
    """Build text with muscle load distribution and period comparison."""
    from metrics import compute_muscle_scores, normalize_scores

    scores_cur = normalize_scores(compute_muscle_scores(df_current, 'rtv'))

    lines = ["\nMuscle load distribution (normalised 0-1):"]
    for m in MUSCLES:
        bar = '█' * int(scores_cur.get(m, 0) * 10)
        lines.append(f"  {m:<15} {scores_cur.get(m, 0):.2f}  {bar}")

    if df_previous is not None and not df_previous.empty:
        scores_prev = normalize_scores(compute_muscle_scores(df_previous, 'rtv'))
        lines.append("\nChange vs previous period:")
        for m in MUSCLES:
            delta = scores_cur.get(m, 0) - scores_prev.get(m, 0)
            arrow = '▲' if delta > 0.05 else ('▼' if delta < -0.05 else '=')
            lines.append(f"  {m:<15} {arrow} {delta:+.2f}")

    return '\n'.join(lines)


# ── Main analysis ─────────────────────────────────────────────────────────────

_FOCUS_INSTRUCTIONS = {
    'general': """\
Identify the 2-3 most meaningful patterns in the recent session data. Comment on
training consistency, load trends, and whether RTV distribution across muscle groups
reflects the stated goal. Flag anything that looks like stagnation or imbalance.""",

    'balance': """\
Analyze the muscle load distribution in detail. Identify which groups are over or
undertrained relative to the others and relative to the stated goal. Distinguish
between structural imbalances (persistent across periods) and recent ones (last 1-2
sessions). Suggest one concrete corrective action.""",

    'progression': """\
For each major exercise in the recent history, assess whether load and RTV are
trending upward, flat, or declining. Flag any exercise where progress has stalled
for 3 or more sessions. Note whether the athlete is progressing faster on compounds
or isolations. Consider set_type context — an AMRAP set with increasing reps_actual
over time is valid evidence of progression even without load increases.""",

    'next_session': """\
Based on the most recent session, the current muscle load distribution, and any
limitations in the persistent notes, recommend a specific approach for the next
training session. Name the day type. For each exercise, suggest whether to maintain,
increase, or reduce load, and whether to push the final set to failure. Be concrete —
give actual numbers where the data supports it.""",
}


def get_llm_analysis(user_profile: dict, df_current, df_previous=None,
                     focus: str = 'general') -> str:
    """
    Generate training analysis via LLM.
    focus: 'general' | 'balance' | 'progression' | 'next_session'
    """
    session_summary = build_session_summary(df_current)
    muscle_summary = build_muscle_summary(df_current, df_previous)
    focus_instruction = _FOCUS_INSTRUCTIONS.get(focus, _FOCUS_INSTRUCTIONS['general'])

    prompt = f"""ATHLETE PROFILE:
- Current body weight: {user_profile.get('bodyweight', 'not available')} kg
- Goal: {user_profile.get('goal', 'not specified')}
- Persistent notes (injuries, limitations, long-term goals): {user_profile.get('memory', 'none')}

RECENT SESSION HISTORY:
{session_summary}

{muscle_summary}

ANALYSIS FOCUS:
{focus_instruction}"""

    return call_llm(prompt)
