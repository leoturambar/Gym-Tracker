import copy
import json
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from config import DAYS, MUSCLES, get_exercise_meta, reload_exercises
from data_manager import (
    init_files, load_sessions, save_session, delete_session,
    get_last_values, get_last_session_meta, save_bodyweight, load_bodyweight,
    get_bodyweight_on, load_memory, save_memory, load_goal, save_goal,
    get_session_dates, import_hevy_rows,
)
from hevy_importer import (
    parse_hevy_csv, group_into_sessions, find_overlap,
    compare_session, sessions_to_csv_rows, load_exercise_map,
    UNASSIGNED_DAY_ID, UNASSIGNED_DAY_NAME,
)
from metrics import (
    compute_muscle_scores, normalize_scores,
    enrich_with_bodyweight, filter_by_period,
    exercise_progression, get_reference_rtv_weekly,
    weekly_muscle_rtv, daily_session_counts, exercise_records, STALL_SESSIONS,
)
from llm import get_llm_analysis, call_llm

# ── Setup ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gym Tracker",
    page_icon="🏋️",
    layout="wide"
)

init_files()
# Rebuild DAYS / exercise metadata from exercises.json on every run, so
# routine sync and new exercises take effect without restarting the app.
reload_exercises()

# ── Module-level constants ────────────────────────────────────────────────────

_EXERCISES_PATH = 'data/exercises.json'
_ALL_SET_TYPES  = ["standard", "amrap", "drop_inverse", "fixed_plus", "none"]
_MUSCLE_KEYS    = ['chest', 'shoulders', 'triceps', 'back', 'biceps',
                   'quads', 'hamstrings', 'core', 'glutes', 'calves']
_MUSCLE_LABELS  = {
    'chest':      'Petto',
    'shoulders':  'Spalle',
    'triceps':    'Tricipiti',
    'back':       'Schiena',
    'biceps':     'Bicipiti',
    'quads':      'Quadricipiti',
    'hamstrings': 'Femorali',
    'core':       'Core',
    'glutes':     'Glutei',
    'calves':     'Polpacci',
}


# Chart colours: validated 8-slot categorical palette, fixed order (one per muscle
# in MUSCLES order, never cycled) + a muted grey for comparison/reference marks.
# Dark mode uses the same hues stepped for the dark surface.
_DARK = getattr(st.context.theme, 'type', None) == 'dark'
_C_SERIES = (['#3987e5', '#d95926', '#199e70', '#c98500',
              '#d55181', '#008300', '#9085e9', '#e66767'] if _DARK else
             ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#6250d6', '#e34948'])
_C_MUTED   = '#898781'
_C_SURFACE = '#0e1117' if _DARK else '#ffffff'   # gap between stacked segments
_C_TEXT2   = '#c3c2b7' if _DARK else '#52514e'   # secondary text on charts
# Calendar colours. Day view is binary (rest / trained: one session per day max).
# Week view is ordinal 0..4+ sessions: grey for rest, then one blue hue with
# rising contrast against the surface.
_C_REST  = '#2a2a28' if _DARK else '#ebeae6'
_C_TRAIN = '#3987e5' if _DARK else '#2a78d6'
_WEEK_STEPS = (['#184f95', '#2a78d6', '#6da7ec', '#b7d3f6'] if _DARK else
               ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])


def _stepped_scale(colors: list) -> list:
    """Plotly colorscale with one flat band per integer value 0..len(colors)-1."""
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    return scale

_HISTORY_PAGE = 15  # sessions shown per page in Storico


def _format_history_entry(row) -> str:
    """One-line Italian summary of a logged exercise for the session history."""
    def _g(col, default=None):
        v = row.get(col, default)
        return default if v is None or (isinstance(v, float) and pd.isna(v)) else v

    ex_type  = row['type']
    sets     = int(_g('sets', 1) or 1)
    reps     = int(_g('reps', 10) or 0)
    set_type = str(_g('set_type', 'standard'))
    value    = float(_g('value', 0.0))
    v2, ra   = _g('value2'), _g('reps_actual')
    vd, rd   = _g('value_drop'), _g('reps_drop')

    if ex_type == 'timed':
        return f"{sets} × {int(value)} s" if sets > 1 else f"{int(value)} s"

    special  = set_type in ('amrap', 'drop_inverse', 'fixed_plus')
    base     = sets - 1 if special and (set_type == 'amrap' and ra is not None
                                        or set_type != 'amrap' and v2 is not None) else sets
    if ex_type == 'bodyweight':
        load = "corpo libero"
    elif ex_type == 'weighted_bw':
        load = (f"assistito {abs(value):g} kg" if value < 0
                else f"zavorra {value:g} kg" if value > 0 else "corpo libero")
    else:
        load = f"{value:g} kg"

    text = f"{load} · {base}×{reps}"
    if set_type == 'amrap' and ra is not None:
        text += f" + AMRAP {int(ra)} rip."
    elif set_type == 'drop_inverse' and v2 is not None:
        text += f" + {float(v2):g} kg × {int(ra) if ra is not None else reps}"
        if vd is not None:
            text += f" → drop {float(vd):g} kg × {int(rd) if rd is not None else reps}"
    elif set_type == 'fixed_plus' and v2 is not None:
        text += f" + ultima a {float(v2):g} kg"
    return text


def _load_ex_file() -> dict:
    """Load exercises.json as {'exercises': [...], 'days': [...]}.
    Handles both old flat-array format and new dict format transparently."""
    if not os.path.exists(_EXERCISES_PATH):
        return {'exercises': [], 'days': []}
    with open(_EXERCISES_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {'exercises': raw, 'days': []}
    return raw


def _save_ex_file(data: dict):
    with open(_EXERCISES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_exercises_json() -> list:
    return _load_ex_file()['exercises']


def _save_exercises_json(exercises: list):
    data = _load_ex_file()
    data['exercises'] = exercises
    _save_ex_file(data)


def _suggest_muscles(name: str, ex_type: str) -> dict | None:
    prompt = (
        f'You are a biomechanics expert. For the exercise "{name}" (type: {ex_type}), '
        'return the primary muscle contributions as a JSON object.\n'
        'Use only these keys: chest, shoulders, triceps, back, biceps, quads, hamstrings, core, glutes, calves.\n'
        'Values must be floats that sum to exactly 1.0. Include only muscles with weight >= 0.05.\n'
        'Return ONLY the JSON object, no explanation, no markdown. Example:\n'
        '{"chest": 0.5, "shoulders": 0.3, "triceps": 0.2}'
    )
    try:
        response = call_llm(prompt, system_prompt='You are a biomechanics expert. Respond with valid JSON only.')
        start    = response.find('{')
        end      = response.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        data     = json.loads(response[start:end])
        filtered = {k: round(float(v), 3) for k, v in data.items()
                    if k in _MUSCLE_KEYS and float(v) >= 0.05}
        return filtered or None
    except Exception:
        return None


def _search_exrx_url(exercise_name: str) -> str | None:
    import re
    try:
        query = f"site:exrx.net/WeightExercises {exercise_name}"
        url   = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        resp  = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        resp.raise_for_status()
        matches = re.findall(r'https?://(?:www\.)?exrx\.net/WeightExercises[^\s"<>&]+', resp.text)
        return matches[0] if matches else None
    except Exception:
        return None


def _fetch_page_text(url: str, max_chars: int = 4000) -> str:
    import re
    try:
        headers = {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection":      "keep-alive",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return ''


def _enrich_from_page(exercise_name: str, page_text: str) -> dict | None:
    prompt = (
        f'From the following ExRx exercise page content for "{exercise_name}", '
        'extract the muscle contributions as a JSON object.\n'
        'Use only these keys: chest, shoulders, triceps, back, biceps, quads, hamstrings, core, glutes, calves.\n'
        'Values must be floats that sum to approximately 1.0. Include only muscles with weight >= 0.05.\n'
        'Return ONLY the JSON object, no explanation.\n\n'
        f'Page content:\n{page_text}'
    )
    try:
        response = call_llm(prompt, system_prompt='You are a biomechanics expert. Respond with valid JSON only.')
        start    = response.find('{')
        end      = response.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        data     = json.loads(response[start:end])
        filtered = {k: round(float(v), 3) for k, v in data.items()
                    if k in _MUSCLE_KEYS and float(v) >= 0.05}
        if not filtered or sum(filtered.values()) < 0.5:
            return None
        return filtered
    except Exception:
        return None


def _padded_range(series: pd.Series, pad: float = 0.15) -> list:
    """Y-axis range with padding that also works for negative (assisted) values."""
    lo, hi = float(series.min()), float(series.max())
    span = hi - lo
    margin = span * pad if span > 0 else max(abs(hi) * pad, 1.0)
    return [lo - margin, hi + margin]


# ── Day metadata helpers ──────────────────────────────────────────────────────

def _get_effective_days() -> list:
    """Returns DAYS with any name overrides from exercises.json applied,
    plus any custom days stored there."""
    data      = _load_ex_file()
    day_meta  = {str(d['id']): d for d in data.get('days', [])}
    days      = copy.deepcopy(DAYS)
    for day in days:
        ov = day_meta.get(str(day['id']), {})
        if ov.get('name'):
            day['name'] = ov['name']
    builtin_ids = {d['id'] for d in DAYS}
    for d in data.get('days', []):
        if d.get('id') not in builtin_ids:
            days.append(copy.deepcopy(d))
    return days


# ── Tab structure ─────────────────────────────────────────────────────────────

tab_allenamento, tab_analisi, tab_profilo = st.tabs([
    "🏋️ Allenamento", "📊 Analisi", "👤 Profilo"
])

with tab_allenamento:

    # ── Part C: Top controls ──────────────────────────────────────────────
    effective_days = _get_effective_days()
    day_labels     = [f"D{d['id']} · {d['name']}" for d in effective_days]

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([4, 1, 1, 1])
    with ctrl_c1:
        sel_day_idx  = st.selectbox(
            "Giorno", range(len(effective_days)),
            format_func=lambda i: day_labels[i],
            key="sel_day_idx", label_visibility="collapsed"
        )
        selected_day = effective_days[sel_day_idx]
    with ctrl_c2:
        if st.button("✏️", key="btn_rename", help="Rinomina giorno"):
            st.session_state['_rename_day_id'] = selected_day['id']
    with ctrl_c3:
        if st.button("➕ Giorno", key="btn_add_day"):
            _exf    = _load_ex_file()
            _new_id = max(d['id'] for d in effective_days) + 1
            _exf.setdefault('days', []).append(
                {'id': _new_id, 'name': f'Giorno {_new_id}', 'sub': '', 'exercises': []}
            )
            _save_ex_file(_exf)
            st.rerun()
    with ctrl_c4:
        session_date     = st.date_input("Data", value=pd.Timestamp.today(),
                                         label_visibility="collapsed")
        session_date_str = str(session_date)

    if st.session_state.get('_rename_day_id') == selected_day['id']:
        new_day_name = st.text_input(
            "Nuovo nome", value=selected_day['name'],
            key="rename_input", label_visibility="collapsed"
        )
        conf_c1, conf_c2 = st.columns([1, 5])
        with conf_c1:
            if st.button("✓", key="btn_confirm_rename"):
                _exf       = _load_ex_file()
                _days_list = _exf.setdefault('days', [])
                _existing  = next((d for d in _days_list
                                   if d.get('id') == selected_day['id']), None)
                if _existing:
                    _existing['name'] = new_day_name
                else:
                    _days_list.append({'id': selected_day['id'], 'name': new_day_name})
                _save_ex_file(_exf)
                st.session_state.pop('_rename_day_id', None)
                st.rerun()
        with conf_c2:
            if st.button("✕", key="btn_cancel_rename"):
                st.session_state.pop('_rename_day_id', None)
                st.rerun()

    st.divider()

    # ── Part D: Exercise cards ────────────────────────────────────────────
    last_values       = get_last_values(selected_day['id'])
    last_session_meta = get_last_session_meta(selected_day['id'])

    all_ex_data  = _load_exercises_json()
    all_ex_names = [e['name'] for e in all_ex_data if e.get('type') != 'excluded']
    all_ex_map   = {e['name']: e for e in all_ex_data}

    day_exercises = [e for e in selected_day.get('exercises', [])
                     if e.get('type') != 'excluded']
    day_id        = selected_day['id']

    _extra_key     = f'_extra_slots_d{day_id}'
    _removed_key   = f'_removed_slots_d{day_id}'
    _confirm_key   = f'_remove_confirm_d{day_id}'
    extra_count    = st.session_state.get(_extra_key, 0)
    _removed_slots = st.session_state.get(_removed_key, set())
    _confirm_slots = st.session_state.get(_confirm_key, set())
    total_slots    = len(day_exercises) + extra_count

    exercise_inputs: dict = {}

    for slot_idx in range(total_slots):
        is_scheduled = slot_idx < len(day_exercises)
        slot_ex      = day_exercises[slot_idx] if is_scheduled else None
        slot_name    = slot_ex['name'] if slot_ex else ''
        slot_type    = slot_ex.get('type', 'weighted') if slot_ex else 'weighted'

        if is_scheduled:
            options = list(dict.fromkeys([slot_name] + all_ex_names))
        else:
            options = [''] + all_ex_names

        if slot_idx in _removed_slots:
            continue

        with st.container(border=True):
            # ── Line 1: exercise selector | skip | remove ─────────────────
            l1c1, l1c2, l1c3 = st.columns([4, 1, 1])
            with l1c1:
                selected_name = st.selectbox(
                    "Esercizio", options, index=0,
                    key=f"ex_d{day_id}s{slot_idx}", label_visibility="collapsed"
                )
                ex_meta     = all_ex_map.get(selected_name, {})
                has_muscles = bool(ex_meta.get('muscles'))
                # ExRx button: exercise is named but has no muscle data in exercises.json
                if selected_name and not has_muscles:
                    if st.button("🔍 Cerca su ExRx", key=f"exrx_d{day_id}s{slot_idx}"):
                        with st.spinner(f"Ricerca {selected_name}…"):
                            _url  = _search_exrx_url(selected_name)
                            _page = _fetch_page_text(_url) if _url else ''
                            _msc  = _enrich_from_page(selected_name, _page) if _page else None
                        if _msc:
                            _exd = _load_exercises_json()
                            _ei  = next((i for i, e in enumerate(_exd)
                                         if e['name'] == selected_name), None)
                            if _ei is not None:
                                _exd[_ei]['muscles'] = _msc
                                _save_exercises_json(_exd)
                                st.success("Muscoli aggiornati.")
                                st.rerun()
                            else:
                                st.warning("Esercizio non trovato — aggiungilo dal Profilo.")
                        else:
                            st.warning("Dati muscolari non trovati su ExRx.")
            with l1c2:
                skipped = st.checkbox("Skip", key=f"skip_d{day_id}s{slot_idx}")
            with l1c3:
                if st.button("✕", key=f"rm_d{day_id}s{slot_idx}", help="Rimuovi dalla sessione"):
                    if is_scheduled:
                        _confirm_slots.add(slot_idx)
                        st.session_state[_confirm_key] = _confirm_slots
                    else:
                        _removed_slots.add(slot_idx)
                        st.session_state[_removed_key] = _removed_slots
                    st.rerun()

            if slot_idx in _confirm_slots:
                st.warning(
                    f"Rimuovere **{slot_name or 'questo esercizio'}** dalla sessione "
                    "e dalla scheda del giorno?"
                )
                conf_rm_c1, conf_rm_c2 = st.columns([1, 5])
                with conf_rm_c1:
                    if st.button("Conferma", key=f"rm_conf_d{day_id}s{slot_idx}", type="primary"):
                        _removed_slots.add(slot_idx)
                        _confirm_slots.discard(slot_idx)
                        st.session_state[_removed_key] = _removed_slots
                        st.session_state[_confirm_key] = _confirm_slots
                        st.rerun()
                with conf_rm_c2:
                    if st.button("Annulla", key=f"rm_cancel_d{day_id}s{slot_idx}"):
                        _confirm_slots.discard(slot_idx)
                        st.session_state[_confirm_key] = _confirm_slots
                        st.rerun()
                continue

            eff_type  = ex_meta.get('type', slot_type)
            last_meta = last_session_meta.get(slot_name, {}) if slot_name else {}

            if skipped:
                exercise_inputs[slot_idx] = {
                    'name': selected_name, 'type': eff_type, 'skipped': True,
                    'value': 0.0, 'sets': 0, 'reps': 0, 'set_type': 'standard',
                    'value2': None, 'reps_actual': None,
                    'value_drop': None, 'reps_drop': None,
                }
                continue

            # ── Column headers ────────────────────────────────────────────
            hdr_a, hdr_b, hdr_c = st.columns([2, 1, 2])
            with hdr_a:
                st.caption("Serie")
            with hdr_b:
                st.caption("Tempo (s)" if eff_type == 'timed' else "Ripetizioni")
            with hdr_c:
                if eff_type not in ('timed', 'bodyweight'):
                    st.caption("Peso (kg)")

            # ── Line 2: sets | reps-or-duration | weight ──────────────────
            c2a, c2b, c2c = st.columns([2, 1, 2])

            with c2a:
                _sets_def = max(1, int((last_meta.get('sets') or ex_meta.get('sets') or 4) if slot_name else 4))
                sets = st.number_input(
                    "Serie", min_value=1, max_value=10, value=_sets_def,
                    step=1, key=f"sets_d{day_id}s{slot_idx}", label_visibility="collapsed"
                )

            with c2b:
                if eff_type == 'timed':
                    _dur_def = float(last_values.get(slot_name) or (slot_ex.get('default', 60) if slot_ex else 60) or 60)
                    value = st.number_input(
                        "Tempo", value=_dur_def, step=5.0,
                        key=f"val_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                    reps = 0
                else:
                    _reps_def = max(1, int((last_meta.get('reps') or ex_meta.get('reps') or 10) if slot_name else 10))
                    reps = st.number_input(
                        "Reps", min_value=1, max_value=50, value=_reps_def,
                        step=1, key=f"reps_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )

            with c2c:
                if eff_type == 'timed':
                    st.empty()  # value already captured in Col B
                elif eff_type == 'bodyweight':
                    value = 0.0
                    st.empty()
                elif eff_type == 'weighted_bw':
                    # No clamp at 0: negative = assisted (e.g. -10 kg on assisted pull-ups)
                    _v_def = float(last_values.get(slot_name) or (slot_ex.get('default', 0) if slot_ex else 0) or 0)
                    value = st.number_input(
                        "kg", value=_v_def, step=1.0,
                        key=f"wval_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                else:
                    _v_def = max(0.0, float(last_values.get(slot_name) or (slot_ex.get('default', 0) if slot_ex else 0) or 0))
                    value = st.number_input(
                        "kg", value=_v_def, step=0.5,
                        key=f"wval_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )

            if eff_type == 'timed':
                exercise_inputs[slot_idx] = {
                    'name': selected_name, 'type': eff_type, 'skipped': False,
                    'value': value, 'sets': int(sets), 'reps': 0, 'set_type': 'none',
                    'value2': None, 'reps_actual': None,
                    'value_drop': None, 'reps_drop': None,
                }
                continue

            # ── Line 3: set type + conditional fields ─────────────────────
            _raw_st    = str(last_meta.get('set_type', ex_meta.get('set_type', 'standard')))
            default_st = _raw_st if _raw_st in _ALL_SET_TYPES else 'standard'

            value2      = None
            reps_actual = None
            value_drop  = None
            reps_drop   = None

            c3a, c3b, c3c = st.columns([2, 1, 2])
            with c3a:
                set_type = st.selectbox(
                    "Tipo", _ALL_SET_TYPES,
                    index=_ALL_SET_TYPES.index(default_st),
                    key=f"st_d{day_id}s{slot_idx}", label_visibility="collapsed"
                )

            if set_type == 'amrap':
                with c3b:
                    _ra_def     = max(1, int(last_meta.get('reps_actual') or reps))
                    reps_actual = st.number_input(
                        "Reps finali", min_value=1, max_value=100,
                        value=_ra_def, step=1,
                        key=f"ra_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                with c3c:
                    _lv2   = last_meta.get('value2')
                    _v2def = float(_lv2) if _lv2 is not None else value
                    value2 = st.number_input(
                        "kg", value=_v2def, step=0.5,
                        key=f"v2_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )

            elif set_type == 'fixed_plus':
                with c3c:
                    _lv2   = last_meta.get('value2')
                    _v2def = float(_lv2) if _lv2 is not None else value + 5.0
                    value2 = st.number_input(
                        "kg", value=_v2def, step=0.5,
                        key=f"v2_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )

            elif set_type == 'drop_inverse':
                # Sub-line 3a: set_type (Col A already) | reps_actual (Col B) | value2 (Col C)
                with c3b:
                    _ra_def     = max(1, int(last_meta.get('reps_actual') or reps))
                    reps_actual = st.number_input(
                        "Reps +peso", min_value=1, max_value=100,
                        value=_ra_def, step=1,
                        key=f"ra_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                with c3c:
                    _lv2   = last_meta.get('value2')
                    _v2def = float(_lv2) if _lv2 is not None else value + 5.0
                    value2 = st.number_input(
                        "kg +peso", value=_v2def, step=0.5,
                        key=f"v2_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                # Sub-line 3b: "Drop" (Col A) | reps_drop (Col B) | value_drop (Col C)
                c3ba, c3bb, c3bc = st.columns([2, 1, 2])
                with c3ba:
                    st.caption("Drop")
                with c3bb:
                    _lrd      = last_meta.get('reps_drop')
                    _rddef    = max(1, int(_lrd) if _lrd is not None else reps)
                    reps_drop = st.number_input(
                        "Reps drop", min_value=1, max_value=100,
                        value=_rddef, step=1,
                        key=f"rd_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )
                with c3bc:
                    _lvd       = last_meta.get('value_drop')
                    _vddef     = float(_lvd) if _lvd is not None else max(0.0, value - 5.0)
                    value_drop = st.number_input(
                        "kg drop", value=_vddef, step=0.5,
                        key=f"vd_d{day_id}s{slot_idx}", label_visibility="collapsed"
                    )

            exercise_inputs[slot_idx] = {
                'name':        selected_name,
                'type':        eff_type,
                'skipped':     False,
                'value':       value,
                'sets':        int(sets),
                'reps':        int(reps),
                'set_type':    set_type,
                'value2':      float(value2) if value2 is not None else None,
                'reps_actual': int(reps_actual) if reps_actual is not None else None,
                'value_drop':  float(value_drop) if value_drop is not None else None,
                'reps_drop':   int(reps_drop) if reps_drop is not None else None,
            }

    # "+ Aggiungi esercizio" — appends a blank card for this session only
    if st.button("+ Aggiungi esercizio", key=f"btn_add_ex_d{day_id}"):
        st.session_state[_extra_key] = extra_count + 1

    # ── Part F: Note + Save ───────────────────────────────────────────────
    note = st.text_area("Note sessione", placeholder="Come ti sei sentito, variazioni, dolori…")

    if st.button("💾 Salva sessione", type="primary"):
        session_id = int(time.time())
        exercises  = []
        for slot_idx in range(total_slots):
            inp = exercise_inputs.get(slot_idx)
            if inp is None or not inp.get('name'):  # skip blank extra cards
                continue
            exercises.append({
                'name':        inp['name'],
                'type':        inp['type'],
                'value':       inp.get('value', 0.0),
                'skipped':     inp.get('skipped', False),
                'variant':     '',
                'sets':        inp.get('sets', 0),
                'reps':        inp.get('reps', 0),
                'set_type':    inp.get('set_type', 'standard'),
                'value2':      inp.get('value2'),
                'reps_actual': inp.get('reps_actual'),
                'value_drop':  inp.get('value_drop'),
                'reps_drop':   inp.get('reps_drop'),
            })
        save_session(session_id, session_date_str,
                     selected_day['id'], selected_day['name'],
                     exercises, note)

        # ── Routine sync ─────────────────────────────────────────────────
        _builtin_ids = {d['id'] for d in DAYS}
        _day_id_str  = f"D{selected_day['id']}"
        _day_ex_set  = {ex['name'] for ex in day_exercises}
        _submitted   = {
            inp['name']: inp.get('set_type', 'standard')
            for inp in exercise_inputs.values()
            if inp and inp.get('name')
        }
        _sub_names  = set(_submitted.keys())
        _added_rt   = _sub_names - _day_ex_set
        _removed_rt = _day_ex_set - _sub_names
        _changed_st = {
            name: _submitted[name]
            for name in _sub_names & _day_ex_set
            if _submitted[name] != all_ex_map.get(name, {}).get('set_type', 'standard')
        }
        if _added_rt or _removed_rt or _changed_st:
            _exf2 = _load_ex_file()
            _exs2 = _exf2.get('exercises', [])
            if selected_day['id'] in _builtin_ids:
                for _n in _added_rt:
                    _ei = next((i for i, e in enumerate(_exs2) if e['name'] == _n), None)
                    if _ei is not None and _day_id_str not in _exs2[_ei].get('day_ids', []):
                        _exs2[_ei].setdefault('day_ids', []).append(_day_id_str)
                for _n in _removed_rt:
                    _ei = next((i for i, e in enumerate(_exs2) if e['name'] == _n), None)
                    if _ei is not None:
                        try:
                            _exs2[_ei]['day_ids'].remove(_day_id_str)
                        except (KeyError, ValueError):
                            pass
                    else:
                        # Hardcoded-only exercise: create a JSON record so the
                        # removal sticks (JSON wins over config.py fallbacks).
                        _slot = next((e for e in day_exercises if e['name'] == _n), {})
                        _other_days = [
                            f"D{d['id']}" for d in DAYS
                            if d['id'] != selected_day['id']
                            and any(e['name'] == _n for e in d['exercises'])
                        ]
                        _rec = {
                            'name':     _n,
                            'type':     _slot.get('type', 'weighted'),
                            'day_ids':  _other_days,
                            'muscles':  {},
                            'set_type': _slot.get('set_type', 'standard'),
                            'no_amrap': _slot.get('no_amrap', False),
                            'variants': _slot.get('variants', []),
                        }
                        if 'default' in _slot:
                            _rec['default'] = _slot['default']
                        _exs2.append(_rec)
            else:
                _days_list2 = _exf2.setdefault('days', [])
                _day_entry  = next(
                    (d for d in _days_list2 if d.get('id') == selected_day['id']), None
                )
                if _day_entry is not None:
                    _day_exs2 = _day_entry.setdefault('exercises', [])
                    for _n in _added_rt:
                        if not any(e['name'] == _n for e in _day_exs2):
                            _em = all_ex_map.get(_n, {})
                            _day_exs2.append({
                                'name':     _n,
                                'type':     _em.get('type', 'weighted'),
                                'set_type': _submitted.get(_n, 'standard'),
                                'no_amrap': _em.get('no_amrap', False),
                                'variants': _em.get('variants', []),
                            })
                    _day_entry['exercises'] = [
                        e for e in _day_exs2 if e['name'] not in _removed_rt
                    ]
            for _n, _new_st in _changed_st.items():
                _ei = next((i for i, e in enumerate(_exs2) if e['name'] == _n), None)
                if _ei is not None:
                    _exs2[_ei]['set_type'] = _new_st
            _exf2['exercises'] = _exs2
            _save_ex_file(_exf2)
            _rt_parts = []
            if _added_rt:
                _rt_parts.append(f"aggiunto {', '.join(sorted(_added_rt))}")
            if _removed_rt:
                _rt_parts.append(f"rimosso {', '.join(sorted(_removed_rt))}")
            if _changed_st:
                _rt_parts.append(f"set type modificato per {', '.join(sorted(_changed_st))}")
            st.info(f"Routine aggiornata: {'; '.join(_rt_parts)}.")

        st.session_state[_extra_key]   = 0
        st.session_state[_removed_key] = set()
        st.session_state[_confirm_key] = set()
        st.success("Sessione salvata!")
        time.sleep(1)
        st.rerun()

    st.divider()

    # ── Part G: Storico sessioni ──────────────────────────────────────────
    st.subheader("Storico sessioni")

    df = load_sessions()

    if df.empty:
        st.info("Nessuna sessione registrata ancora.")
    else:
        sessions_meta = (
            df[['session_id', 'date', 'day_name', 'note']]
            .drop_duplicates(subset='session_id')
            .sort_values(['date', 'session_id'], ascending=False)
        )
        _hist_limit = st.session_state.get('_hist_limit', _HISTORY_PAGE)

        for _, smeta in sessions_meta.head(_hist_limit).iterrows():
            sid      = smeta['session_id']
            date     = smeta['date']
            day_name = smeta['day_name']
            note     = smeta['note'] if pd.notna(smeta['note']) else ''

            with st.expander(f"**{date}** — {day_name}"):
                sess_df = df[df['session_id'] == sid]
                for _, row in sess_df.iterrows():
                    if row['skipped']:
                        st.write(f"~~{row['exercise']}~~ — saltato")
                    elif row['type'] == 'excluded':
                        continue
                    else:
                        st.write(f"**{row['exercise']}**: {_format_history_entry(row)}")

                if note:
                    st.caption(f"📝 {note}")

                if st.button("🗑️ Elimina sessione", key=f"del_{sid}"):
                    delete_session(sid)
                    st.rerun()

        _n_total = len(sessions_meta)
        if _n_total > _hist_limit:
            if st.button(f"Mostra altre (visualizzate {_hist_limit} di {_n_total})",
                         key="btn_hist_more"):
                st.session_state['_hist_limit'] = _hist_limit + _HISTORY_PAGE
                st.rerun()

with tab_analisi:

    df_all = load_sessions()
    if not df_all.empty:
        df_all = enrich_with_bodyweight(df_all)

    # ── Bilancio muscolare ────────────────────────────────────────────────
    st.subheader("Bilancio muscolare")

    if df_all.empty:
        st.info("Nessuna sessione registrata ancora.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            metric = st.radio(
                "Metrica",
                options=['rtv', 'freq'],
                format_func=lambda x: 'Carico relativo (RTV)' if x == 'rtv' else 'Frequenza esercizi',
                horizontal=True
            )
        with col2:
            period = st.radio(
                "Confronta",
                options=['week', 'month', 'year'],
                format_func=lambda x: {
                    'week':  'vs Sett. prec.',
                    'month': 'vs Mese prec.',
                    'year':  'vs Anno prec.'
                }[x],
                horizontal=True
            )

        show_reference = st.checkbox(
            "Confronta con l'atleta di riferimento (% per muscolo)", value=False,
            disabled=(metric != 'rtv'),
            help="Disponibile solo con la metrica RTV.",
        )
        show_reference = show_reference and metric == 'rtv'

        if metric == 'rtv' and df_all['bodyweight'].isna().all():
            st.warning("Imposta il peso corporeo nel tab Profilo per usare la metrica RTV.")

        _today_n = pd.Timestamp.today().normalize()
        if period == 'week':
            cur_weeks  = 1.0
            prev_weeks = 1.0
        elif period == 'month':
            _start_cur_m  = _today_n.replace(day=1)
            cur_weeks     = max(((_today_n - _start_cur_m).days + 1) / 7.0, 1 / 7)
            _start_prev_m = (_start_cur_m - pd.Timedelta(days=1)).replace(day=1)
            prev_weeks    = max((_start_cur_m - _start_prev_m).days / 7.0, 1 / 7)
        else:  # year
            _start_cur_y  = _today_n.replace(month=1, day=1)
            cur_weeks     = max(((_today_n - _start_cur_y).days + 1) / 7.0, 1 / 7)
            _start_prev_y = _start_cur_y.replace(year=_start_cur_y.year - 1)
            prev_weeks    = max((_start_cur_y - _start_prev_y).days / 7.0, 1 / 7)

        df_cur, df_prev = filter_by_period(df_all, period)
        scores_a = compute_muscle_scores(df_cur, metric, period_weeks=cur_weeks)
        scores_b = compute_muscle_scores(df_prev, metric, period_weeks=prev_weeks)

        # Scores are already RTV per week, so the weekly reference is directly
        # comparable for every period — no scaling by period length.
        _ref_weekly = get_reference_rtv_weekly()

        all_values = list(scores_a.values()) + list(scores_b.values())
        max_val  = max(all_values) if any(all_values) else 1
        axis_max = max_val * 1.15

        categories = MUSCLES + [MUSCLES[0]]
        values_a = [scores_a.get(m, 0) for m in MUSCLES] + [scores_a.get(MUSCLES[0], 0)]
        values_b = [scores_b.get(m, 0) for m in MUSCLES] + [scores_b.get(MUSCLES[0], 0)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_b, theta=categories,
            fill='toself', name='Precedente',
            hovertemplate="%{theta}: %{r:.2f}<extra>Precedente</extra>",
            line=dict(color=_C_MUTED, dash='dash', width=2),
            fillcolor='rgba(137,135,129,0.10)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_a, theta=categories,
            fill='toself', name='Corrente',
            hovertemplate="%{theta}: %{r:.2f}<extra>Corrente</extra>",
            line=dict(color=_C_SERIES[0], width=2),
            fillcolor='rgba(42,120,214,0.15)'
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(240,240,240,0.3)',
                radialaxis=dict(
                    visible=True,
                    range=[0, axis_max],
                    tickfont=dict(size=11, color='#666666'),
                    gridcolor='rgba(150,150,150,0.4)',
                    linecolor='rgba(150,150,150,0.4)',
                ),
                angularaxis=dict(
                    tickfont=dict(size=13),
                    gridcolor='rgba(150,150,150,0.3)',
                    linecolor='rgba(150,150,150,0.4)',
                )
            ),
            showlegend=True,
            height=500,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # ── Radar (75%) + bar list (25%) side by side ─────────────────────
        radar_col, bar_col = st.columns([3, 1])
        with radar_col:
            st.plotly_chart(fig, width='stretch')
        with bar_col:
            if show_reference:
                st.caption("% dell'atleta di riferimento (RTV / sett.)")
                for m in MUSCLES:
                    _ref = _ref_weekly.get(m, 0)
                    if _ref > 0:
                        _pct = scores_a.get(m, 0) / _ref
                        st.progress(min(_pct, 1.0), text=f"{m}: {_pct:.0%}")
                    else:
                        st.progress(0.0, text=f"{m}: — (nessun riferimento)")
                st.caption("100% = volume settimanale dell'atleta di riferimento (75 kg).")
            else:
                st.caption(f"{'RTV / sett.' if metric == 'rtv' else 'esercizi / sett.'}")
                _max_bar = max(scores_a.values()) if any(scores_a.values()) else 1
                for m in MUSCLES:
                    _v = scores_a.get(m, 0)
                    st.progress(_v / _max_bar if _max_bar > 0 else 0, text=f"{m}: {_v:.2f}")

        # ── Trend settimanale RTV per muscolo ─────────────────────────────
        st.subheader("Trend settimanale")
        _n_weeks = st.radio(
            "Finestra", options=[12, 26, 52], horizontal=True,
            format_func=lambda n: f"{n} settimane", key="trend_weeks",
        )
        _weekly = weekly_muscle_rtv(df_all, n_weeks=_n_weeks)
        _x = [d.strftime('%d/%m') for d in _weekly.index]
        _x[-1] = _x[-1] + ' (in corso)'

        _fig_tr = go.Figure()
        for i, m in enumerate(MUSCLES):
            _fig_tr.add_trace(go.Bar(
                x=_x, y=_weekly[m], name=m,
                marker=dict(color=_C_SERIES[i], line=dict(color=_C_SURFACE, width=1)),
                hovertemplate=f"{m}: %{{y:.2f}}<extra></extra>",
            ))
        _ref_total = sum(_ref_weekly.values())
        if _ref_total > 0:
            _fig_tr.add_hline(
                y=_ref_total, line=dict(color=_C_MUTED, dash='dot', width=2),
                annotation_text=f"Riferimento {_ref_total:.1f}",
                annotation_position='top left',
                annotation_font=dict(color=_C_TEXT2, size=11),
            )
        _fig_tr.update_layout(
            barmode='stack', bargap=0.25, height=380,
            hovermode='x unified',
            yaxis=dict(title='RTV / settimana', gridcolor='rgba(150,150,150,0.25)'),
            xaxis=dict(title='Settimana dal', tickangle=-45 if _n_weeks > 12 else 0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, traceorder='normal'),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_tr, width='stretch')
        st.caption("Ogni barra è una settimana (lunedì–domenica); l'ultima è quella in corso, "
                   "quindi parziale. La linea tratteggiata è il volume totale "
                   "dell'atleta di riferimento.")
        with st.expander("Tabella trend"):
            _tbl = _weekly.copy()
            _tbl.index = _x
            _tbl['Totale'] = _tbl.sum(axis=1)
            st.dataframe(_tbl.round(2).iloc[::-1], width='stretch')

        # ── Costanza ──────────────────────────────────────────────────────
        st.subheader("Costanza")
        _counts = daily_session_counts(df_all)
        _cal_weeks = 26
        _monday = _today_n - pd.Timedelta(days=_today_n.weekday())
        _start = _monday - pd.Timedelta(weeks=_cal_weeks - 1)
        _days = pd.date_range(_start, _monday + pd.Timedelta(days=6), freq='D')
        _weekdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
        _z = [[None] * _cal_weeks for _ in range(7)]
        _txt = [[''] * _cal_weeks for _ in range(7)]
        for _d in _days:
            _w = (_d - _start).days // 7
            _wd = _d.weekday()
            if _d > _today_n:
                continue
            _n = int(_counts.get(_d.strftime('%Y-%m-%d'), 0))
            _z[_wd][_w] = min(_n, 2)
            _txt[_wd][_w] = (f"{_d.strftime('%d/%m/%Y')}: "
                             f"{_n} session{'e' if _n == 1 else 'i'}")
        _week_labels = [(_start + pd.Timedelta(weeks=i)).strftime('%d/%m')
                        for i in range(_cal_weeks)]

        _sess_per_week = []
        for i in range(_cal_weeks):
            _ws = _start + pd.Timedelta(weeks=i)
            _we = _ws + pd.Timedelta(days=6)
            _sess_per_week.append(int(sum(
                v for k, v in _counts.items()
                if _ws <= pd.Timestamp(k) <= _we
            )))
        _streak = 0
        # current week may still be empty: start counting from last week then
        _seq = _sess_per_week[:-1] if _sess_per_week[-1] == 0 else _sess_per_week
        for v in reversed(_seq):
            if v == 0:
                break
            _streak += 1
        _last4 = sum(_sess_per_week[-4:])
        _avg12 = sum(_sess_per_week[-12:]) / 12

        k1, k2, k3 = st.columns(3)
        k1.metric("Sessioni ultime 4 settimane", _last4)
        k2.metric("Media sessioni / sett. (12 sett.)", f"{_avg12:.1f}")
        k3.metric("Settimane consecutive attive", _streak)

        _cal_view = st.radio(
            "Vista", options=['days', 'weeks'], horizontal=True, key="cal_view",
            format_func=lambda v: {'days':  'Giorni (ultime 26 settimane)',
                                   'weeks': "Settimane dell'anno"}[v],
        )

        if _cal_view == 'days':
            _fig_cal = go.Figure(go.Heatmap(
                z=[[None if v is None else min(v, 1) for v in row] for row in _z],
                x=_week_labels, y=_weekdays, text=_txt,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0.0, _C_REST], [0.5, _C_REST],
                            [0.5, _C_TRAIN], [1.0, _C_TRAIN]],
                zmin=0, zmax=1, xgap=3, ygap=3, showscale=False,
            ))
            _fig_cal.update_layout(
                height=230, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange='reversed', showgrid=False),
                xaxis=dict(showgrid=False, tickangle=-45, nticks=13),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(_fig_cal, width='stretch')
            st.caption("Ogni casella è un giorno: grigio = riposo, blu = allenamento. "
                       "Colonne = settimane (dal lunedì indicato).")
        else:
            # One cell per ISO week: rows = years (newest on top), columns = week 1..53
            _first = pd.Timestamp(min(_counts.index)) if len(_counts) else _today_n
            _first_monday = _first - pd.Timedelta(days=_first.weekday())
            _iso_now = _today_n.isocalendar()
            _years = list(range(_first.isocalendar()[0], _iso_now[0] + 1))
            _per_week: dict = {}
            for _k, _v in _counts.items():
                _iy, _iw, _ = pd.Timestamp(_k).isocalendar()
                _per_week[(_iy, _iw)] = _per_week.get((_iy, _iw), 0) + int(_v)

            _cap = len(_WEEK_STEPS)          # colour cap: 4+ sessions share the top step
            _zw, _tw = [], []
            for _y in _years:
                _row_z, _row_t = [], []
                for _w in range(1, 54):
                    try:
                        _ws = pd.Timestamp.fromisocalendar(_y, _w, 1)
                    except ValueError:       # year without ISO week 53
                        _row_z.append(None); _row_t.append('')
                        continue
                    if _ws > _today_n or _ws < _first_monday:
                        _row_z.append(None); _row_t.append('')
                        continue
                    _n = _per_week.get((_y, _w), 0)
                    _row_z.append(min(_n, _cap))
                    _we = _ws + pd.Timedelta(days=6)
                    _row_t.append(f"{_y} · sett. {_w} ({_ws.strftime('%d/%m')}–"
                                  f"{_we.strftime('%d/%m')}): {_n} session"
                                  f"{'e' if _n == 1 else 'i'}")
                _zw.append(_row_z); _tw.append(_row_t)

            _colors = [_C_REST] + _WEEK_STEPS
            _fig_wk = go.Figure(go.Heatmap(
                z=_zw, x=list(range(1, 54)), y=[str(y) for y in _years], text=_tw,
                hovertemplate="%{text}<extra></extra>",
                colorscale=_stepped_scale(_colors),
                zmin=-0.5, zmax=_cap + 0.5, xgap=3, ygap=3,
                colorbar=dict(
                    title=dict(text='Sessioni', side='right'),
                    tickvals=list(range(_cap + 1)),
                    ticktext=[str(i) for i in range(_cap)] + [f"{_cap}+"],
                    thickness=12, len=1.0, outlinewidth=0,
                ),
            ))
            _fig_wk.update_layout(
                # category axis draws the first year at the bottom → newest on top
                height=110 + 44 * len(_years), margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(showgrid=False, type='category'),
                xaxis=dict(showgrid=False, title='Settimana', dtick=4),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(_fig_wk, width='stretch')
            st.caption("Ogni casella è una settimana dell'anno (lunedì–domenica): "
                       "più è intenso il blu, più sessioni hai fatto. "
                       "Passa sopra una casella per le date.")

    st.divider()

    # ── Progressione ──────────────────────────────────────────────────────
    st.subheader("Progressione")

    if df_all.empty:
        st.info("Nessun dato ancora.")
    else:
        _records = exercise_records(df_all)
        if not _records.empty:
            st.markdown("**Record e stalli**")
            _rec = _records.sort_values(['since_pr', 'exercise'], ascending=[False, True])

            def _status(n: int) -> str:
                if n >= STALL_SESSIONS:
                    return f"⚠️ Fermo da {n} sessioni"
                if n == 0:
                    return "🏆 Record all'ultima sessione"
                return f"✅ Record {n} session{'e' if n == 1 else 'i'} fa"

            st.dataframe(
                pd.DataFrame({
                    'Esercizio':      _rec['exercise'],
                    'Stato':          _rec['since_pr'].apply(_status),
                    'Sessioni':       _rec['sessions'],
                    'Carico max (kg)': _rec['max_load'],
                    '1RM stimato (kg)': _rec['best_e1rm'],
                    'Data record':    _rec['pr_date'],
                    'Ultima volta':   _rec['last_date'],
                }),
                hide_index=True, width='stretch',
                column_config={
                    'Carico max (kg)':  st.column_config.NumberColumn(format="%.1f"),
                    '1RM stimato (kg)': st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                f"Record = miglior 1RM stimato (Epley; include AMRAP e serie finali più "
                f"pesanti). «Fermo» = {STALL_SESSIONS}+ sessioni senza superarlo. "
                "Per gli esercizi con zavorra/assistenza il 1RM include il peso corporeo."
            )

        _prog_names = set(
            ex['name']
            for day in DAYS
            for ex in day['exercises']
            if ex['type'] in ('weighted', 'weighted_bw')
        )
        _logged = sorted(_records['exercise']) if not _records.empty else []
        # Exercises with logged data first, then the rest of the routine
        weighted_exercises = _logged + sorted(_prog_names - set(_logged))

        selected_ex = st.selectbox("Esercizio", weighted_exercises)
        df_prog     = exercise_progression(selected_ex, df_all)

        if df_prog.empty:
            st.info("Nessun dato per questo esercizio ancora.")
        else:
            _has_e1rm = df_prog['e1rm'].notna().any()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sessioni", len(df_prog))
            with col2:
                st.metric("Carico attuale", f"{df_prog['value'].iloc[-1]:.1f} kg",
                          delta=f"{df_prog['value'].iloc[-1] - df_prog['value'].iloc[0]:+.1f} kg dall'inizio",
                          delta_color='off')
            with col3:
                st.metric("Carico massimo", f"{df_prog['value'].max():.1f} kg")
            with col4:
                if _has_e1rm:
                    _e = df_prog['e1rm'].dropna()
                    st.metric("1RM stimato", f"{_e.iloc[-1]:.1f} kg",
                              delta=f"{_e.iloc[-1] - _e.iloc[0]:+.1f} kg dall'inizio")
                else:
                    st.metric("1RM stimato", "—")

            _ex_types = df_all.loc[df_all['exercise'] == selected_ex, 'type']
            _is_weighted = (not _ex_types.empty) and _ex_types.iloc[-1] == 'weighted'

            prog_left, prog_right = st.columns([1, 1])
            with prog_left:
                if _has_e1rm and _is_weighted:
                    st.markdown("**Carico e 1RM stimato (kg)**")
                else:
                    st.markdown("**Carico (kg)**")
                _fig_kg = go.Figure()
                _fig_kg.add_trace(go.Scatter(
                    x=df_prog['date'], y=df_prog['value'],
                    mode='lines+markers', name='Carico di lavoro',
                    line=dict(color=_C_SERIES[0], width=2), marker=dict(size=8),
                    hovertemplate="%{x}: %{y:.1f} kg<extra>Carico</extra>",
                ))
                _y_all = df_prog['value']
                if _has_e1rm and _is_weighted:
                    _fig_kg.add_trace(go.Scatter(
                        x=df_prog['date'], y=df_prog['e1rm'],
                        mode='lines+markers', name='1RM stimato',
                        line=dict(color=_C_SERIES[1], width=2, dash='dot'),
                        marker=dict(size=8),
                        hovertemplate="%{x}: %{y:.1f} kg<extra>1RM stimato</extra>",
                    ))
                    _y_all = pd.concat([_y_all, df_prog['e1rm'].dropna()])
                _fig_kg.update_layout(
                    yaxis=dict(range=_padded_range(_y_all)),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                    hovermode='x unified',
                    margin=dict(l=0, r=0, t=30, b=0), height=300,
                )
                st.plotly_chart(_fig_kg, width='stretch')
            with prog_right:
                if df_prog['rtv'].sum() > 0:
                    st.markdown("**RTV nel tempo**")
                    _fig_rtv = go.Figure()
                    _fig_rtv.add_trace(go.Scatter(
                        x=df_prog['date'], y=df_prog['rtv'],
                        mode='lines+markers', name='RTV',
                        line=dict(color=_C_SERIES[0], width=2), marker=dict(size=8),
                        hovertemplate="%{x}: %{y:.2f}<extra>RTV</extra>",
                    ))
                    _fig_rtv.update_layout(
                        yaxis=dict(range=_padded_range(df_prog['rtv'])),
                        margin=dict(l=0, r=0, t=30, b=0), height=300,
                    )
                    st.plotly_chart(_fig_rtv, width='stretch')
                else:
                    st.caption("Imposta il peso corporeo nel tab Profilo per vedere il grafico RTV.")

    st.divider()

    # ── Analisi AI ────────────────────────────────────────────────────────
    st.subheader("Analisi AI")

    if df_all.empty:
        st.info("Logga almeno una sessione per usare l'analisi AI.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            period_llm = st.radio(
                "Periodo",
                options=['all', 'week', 'month', 'year'],
                format_func=lambda x: {
                    'all':   'Tutto',
                    'week':  'Questa settimana',
                    'month': 'Questo mese',
                    'year':  "Quest'anno",
                }[x],
                horizontal=True
            )
        with col2:
            focus = st.radio(
                "Focus",
                options=['general', 'balance', 'progression', 'next_session'],
                format_func=lambda x: {
                    'general':      'Analisi generale',
                    'balance':      'Equilibrio muscolare',
                    'progression':  'Progressione carichi',
                    'next_session': 'Prossima sessione',
                }[x],
                horizontal=True
            )

        if period_llm == 'all':
            df_cur_ai  = df_all
            df_prev_ai = None
        else:
            df_cur_ai, df_prev_ai = filter_by_period(df_all, period_llm)

        current_bw   = get_bodyweight_on(str(pd.Timestamp.today().date()))
        user_profile = {
            'bodyweight': current_bw,
            'goal':       load_goal() or 'non specificato',
            'memory':     load_memory(),
        }

        st.divider()

        if st.button("🤖 Genera analisi", type="primary"):
            with st.spinner("Patches O'Houlihan sta analizzando i tuoi allenamenti…"):
                try:
                    analysis = get_llm_analysis(user_profile, df_cur_ai, df_prev_ai, focus)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Errore API: {e}")

with tab_profilo:
    st.header("Profilo")

    # ── Peso corporeo ─────────────────────────────────────────────────────
    st.subheader("Peso corporeo")

    df_bw = load_bodyweight()

    col1, col2 = st.columns(2)
    with col1:
        bw_date = st.date_input("Data misurazione", value=pd.Timestamp.today(), key="bw_date")
        df_bw_current = load_bodyweight()
        last_bw = float(df_bw_current.sort_values('date', ascending=False).iloc[0]['bodyweight']) if not df_bw_current.empty else 75.0
        bw_value = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, step=0.1, value=last_bw, key="bw_value")
        if st.button("💾 Salva peso", type="primary"):
            save_bodyweight(bw_value, str(bw_date))
            st.success(f"Peso {bw_value} kg salvato per il {bw_date}.")
            st.rerun()

    with col2:
        if not df_bw.empty:
            _bw_plot = df_bw.dropna().copy()
            _bw_plot['date'] = pd.to_datetime(_bw_plot['date'])
            _bw_plot = _bw_plot.sort_values('date')
            st.markdown("**Peso corporeo nel tempo**")
            _fig_bw = go.Figure(go.Scatter(
                x=_bw_plot['date'], y=_bw_plot['bodyweight'],
                mode='lines+markers', name='Peso',
                line=dict(color=_C_SERIES[0], width=2), marker=dict(size=8),
                hovertemplate="%{x|%d/%m/%Y}: %{y:.1f} kg<extra></extra>",
            ))
            _fig_bw.update_layout(
                yaxis=dict(range=_padded_range(_bw_plot['bodyweight']),
                           title='kg'),
                margin=dict(l=0, r=0, t=10, b=0), height=260,
            )
            st.plotly_chart(_fig_bw, width='stretch')
            with st.expander("Storico peso corporeo"):
                st.dataframe(
                    df_bw.sort_values('date', ascending=False).reset_index(drop=True),
                    width='stretch',
                    hide_index=True
                )
        else:
            st.info("Nessuna misurazione ancora.")

    st.divider()

    # ── Obiettivo e note persistenti ──────────────────────────────────────
    st.subheader("Note persistenti")
    st.caption("Infortuni cronici, limitazioni, obiettivi a lungo termine. Claude le leggerà sempre.")

    current_memory = load_memory()
    new_memory = st.text_area(
        "Note",
        value=current_memory,
        height=150,
        placeholder="Es: spalla sinistra fragile, evitare sovraccarico overhead. Obiettivo: trazioni non assistite entro 6 mesi."
    )

    if st.button("💾 Salva note", type="primary"):
        save_memory(new_memory)
        st.success("Note salvate.")

    st.divider()

    # ── Obiettivo allenamento ─────────────────────────────────────────────
    st.subheader("Obiettivo")
    goal_options = [
        "Ipertrofia (aumento massa muscolare)",
        "Forza (aumento carichi)",
        "Resistenza muscolare",
        "Ricomposizione corporea",
        "Mantenimento",
    ]
    saved_goal = load_goal()
    goal_index = goal_options.index(saved_goal) if saved_goal in goal_options else 0
    goal = st.selectbox("Obiettivo principale", goal_options, index=goal_index)
    if goal != saved_goal:
        save_goal(goal)
    st.session_state['goal'] = goal

    st.divider()

    # ── Gestione esercizi ──────────────────────────────────────────────────
    with st.expander("Gestione esercizi"):

        # Section 1 — Aggiungi esercizio manualmente
        st.subheader("Aggiungi esercizio manualmente")

        ex_name_new = st.text_input("Nome esercizio", key="new_ex_name")
        ex_type_new = st.selectbox(
            "Tipo", ["weighted", "bodyweight", "weighted_bw", "timed", "excluded"],
            key="new_ex_type"
        )

        _eff_days       = _get_effective_days()
        _day_labels     = [f"D{d['id']} - {d['name']}" for d in _eff_days]
        ex_days_labels  = st.multiselect("Giorni", _day_labels, key="new_ex_days")
        ex_days_new     = [lbl.split(' - ')[0] for lbl in ex_days_labels]

        if ex_type_new in ("weighted", "weighted_bw"):
            ex_default_new = st.number_input("Carico default (kg)", value=20.0, step=0.5,
                                             key="new_ex_default")
        else:
            ex_default_new = None

        col_ss, col_rs = st.columns(2)
        with col_ss:
            ex_sets_new = st.number_input("Serie default", min_value=1, max_value=10, value=4,
                                          step=1, key="new_ex_sets")
        with col_rs:
            ex_reps_new = st.number_input("Reps default", min_value=1, max_value=50, value=10,
                                          step=1, key="new_ex_reps")

        ex_set_type_new = st.selectbox("Tipo serie default", _ALL_SET_TYPES, key="new_ex_set_type")
        ex_no_amrap_new = st.checkbox("No AMRAP (disabilita failure sets)", key="new_ex_no_amrap")

        if st.button("✨ Suggerisci muscoli con AI", disabled=not ex_name_new):
            with st.spinner("Calcolo contributi muscolari..."):
                suggested = _suggest_muscles(ex_name_new, ex_type_new)
            if suggested:
                st.session_state['_new_ex_suggested'] = suggested
                for k in _MUSCLE_KEYS:
                    st.session_state[f"mw_{k}"] = round(float(suggested.get(k, 0.0)), 2)
                st.rerun()
            else:
                st.warning("Impossibile ottenere suggerimenti. Inserisci i pesi manualmente.")

        st.write("**Contributi muscolari** (somma consigliata ≈ 1.0)")
        muscle_weights = {}
        mw_cols = st.columns(2)
        for i, mkey in enumerate(_MUSCLE_KEYS):
            with mw_cols[i % 2]:
                muscle_weights[mkey] = st.number_input(
                    _MUSCLE_LABELS[mkey], min_value=0.0, max_value=1.0,
                    value=0.0, step=0.05, key=f"mw_{mkey}"
                )

        total_w = sum(muscle_weights.values())
        if total_w > 0 and abs(total_w - 1.0) >= 0.05:
            st.warning(f"Somma pesi: {total_w:.2f} — deve essere 1.0")
        else:
            w_color = "green" if abs(total_w - 1.0) < 0.05 else "gray"
            st.caption(f"Somma pesi: :{w_color}[{total_w:.2f}]")

        if st.button("💾 Salva esercizio", type="primary"):
            if not ex_name_new:
                st.error("Inserisci il nome dell'esercizio.")
            elif not ex_days_new:
                st.error("Seleziona almeno un giorno.")
            else:
                muscles_dict = {k: v for k, v in muscle_weights.items() if v > 0}
                new_ex: dict = {
                    'name':     ex_name_new,
                    'type':     ex_type_new,
                    'day_ids':  ex_days_new,
                    'muscles':  muscles_dict,
                    'set_type': ex_set_type_new,
                    'no_amrap': bool(ex_no_amrap_new),
                    'variants': [],
                    'sets':     int(ex_sets_new),
                    'reps':     int(ex_reps_new),
                }
                if ex_default_new is not None:
                    new_ex['default'] = float(ex_default_new)

                exercises_data = _load_exercises_json()
                idx = next((i for i, e in enumerate(exercises_data)
                            if e['name'] == ex_name_new), None)
                if idx is not None:
                    exercises_data[idx] = new_ex
                else:
                    exercises_data.append(new_ex)
                _save_exercises_json(exercises_data)
                reload_exercises()

                for k in (['new_ex_name', 'new_ex_days', '_new_ex_suggested']
                          + [f"mw_{m}" for m in _MUSCLE_KEYS]):
                    st.session_state.pop(k, None)

                st.success(f"Esercizio '{ex_name_new}' salvato!")
                st.rerun()

        st.divider()

        # Section 2 — Arricchisci da ExRx (manuale)
        with st.expander("Strumenti avanzati — ExRx enrichment"):
            st.write("**Arricchisci dati muscolari da ExRx**")
            st.caption("Seleziona un esercizio, incolla l'URL ExRx corrispondente e avvia l'estrazione.")

            _exrx_data = _load_exercises_json()

            if not _exrx_data:
                st.info("Nessun esercizio in exercises.json.")
            else:
                _exrx_options = [
                    f"{e['name']} ({e.get('source', 'manual')})"
                    for e in _exrx_data
                ]
                _exrx_sel_idx = st.selectbox(
                    "Esercizio", range(len(_exrx_options)),
                    format_func=lambda i: _exrx_options[i],
                    key="exrx_sel_idx"
                )
                _exrx_ex = _exrx_data[_exrx_sel_idx]

                _exrx_url_default = _exrx_ex.get('exrx_url', '')
                _exrx_url = st.text_input(
                    "ExRx URL", value=_exrx_url_default,
                    placeholder="https://exrx.net/WeightExercises/...",
                    key="exrx_url_input"
                )

                if st.button("🔍 Arricchisci", disabled=not _exrx_url):
                    with st.spinner(f"Fetching {_exrx_url}…"):
                        _page = _fetch_page_text(_exrx_url)
                    if not _page:
                        st.error("Impossibile scaricare la pagina. Verifica l'URL.")
                    else:
                        with st.spinner("Estrazione muscoli via AI…"):
                            _muscles = _enrich_from_page(_exrx_ex['name'], _page)
                        _total = sum(_muscles.values()) if _muscles else 0.0
                        if _muscles and 0.95 <= _total <= 1.05:
                            _exrx_ex['muscles']  = _muscles
                            _exrx_ex['source']   = 'exrx'
                            _exrx_ex['exrx_url'] = _exrx_url
                            _save_exercises_json(_exrx_data)
                            st.success(
                                "Muscoli aggiornati: "
                                + ", ".join(f"{k}={v:.2f}" for k, v in _muscles.items())
                            )
                            st.rerun()
                        else:
                            _raw = str(_muscles) if _muscles else "(nessun risultato)"
                            st.error(
                                f"Risultato non valido (somma={_total:.2f}). "
                                f"Risposta grezza: {_raw}"
                            )

                st.divider()
                st.caption("**Stato arricchimento**")
                _status_rows = [
                    {
                        'Esercizio': e['name'],
                        'Fonte':     e.get('source', '—'),
                        'ExRx URL':  e.get('exrx_url', ''),
                    }
                    for e in _exrx_data
                ]
                st.dataframe(
                    pd.DataFrame(_status_rows),
                    hide_index=True,
                    width='stretch',
                )

    st.divider()

    # ── Importa da Hevy ───────────────────────────────────────────────────
    with st.expander("📥 Importa da Hevy"):
        _hevy_file = st.file_uploader(
            "Carica CSV Hevy", type="csv", key="hevy_uploader"
        )

        if _hevy_file is not None:
            try:
                _df_raw      = parse_hevy_csv(_hevy_file)
                _sessions    = group_into_sessions(_df_raw)
                _ex_dates    = get_session_dates()
                _exist_sess, _new_sess = find_overlap(_sessions, _ex_dates)

                _X = len(_sessions)
                _Y = len(_new_sess)
                _Z = len(_exist_sess)

                st.markdown(
                    f"**{_X} sessioni trovate · {_Y} nuove · {_Z} già presenti**"
                )

                _unassigned = [s for s in _new_sess if s['day_id'] == UNASSIGNED_DAY_ID]
                if _unassigned:
                    st.warning(
                        f"{len(_unassigned)} sessioni senza prefisso D1–D4 nel titolo: "
                        f"verranno importate come «{UNASSIGNED_DAY_NAME}» (contano "
                        "nell'analisi, ma non precompilano nessun giorno). "
                        "La Leg press di queste sessioni viene saltata, perché la "
                        "variante dipende dal giorno.\n\n"
                        + "\n".join(f"- {s['date']} · {s['title']}" for s in _unassigned)
                    )

                # ── Step 2: session overview ──────────────────────────────
                _ov_left, _ov_right = st.columns(2)

                with _ov_left:
                    st.markdown(f"**:green[Nuove ({_Y})]**")
                    for _s in _new_sess:
                        _n_ex = len({
                            e['app_name']
                            for e in _s['exercises']
                            if e['app_name'] and not e['is_warmup']
                        })
                        st.write(f"• {_s['date']} · {_s['day_id']} · {_n_ex} esercizi")

                # Load existing session rows once for all comparisons
                _app_df = load_sessions()

                with _ov_right:
                    st.markdown(f"**:orange[Già presenti ({_Z})]**")
                    for _s in _exist_sess:
                        _app_rows = (
                            _app_df[_app_df['date'] == _s['date']]
                            if not _app_df.empty else pd.DataFrame()
                        )
                        _cmp = compare_session(_s, _app_rows)
                        _badge = "✓" if _cmp['is_clean'] else "⚠"
                        st.write(f"{_badge} {_s['date']} · {_s['day_id']}")

                # ── Step 3: comparison detail for sessions with differences ─
                for _s in _exist_sess:
                    _app_rows = (
                        _app_df[_app_df['date'] == _s['date']]
                        if not _app_df.empty else pd.DataFrame()
                    )
                    _cmp = compare_session(_s, _app_rows)
                    if _cmp['is_clean']:
                        continue
                    _n_diffs = (
                        len(_cmp['only_in_hevy'])
                        + len(_cmp['only_in_app'])
                        + len(_cmp['load_diffs'])
                    )
                    with st.expander(
                        f"{_s['date']} {_s['day_id']} — {_n_diffs} differenze",
                        expanded=False,
                    ):
                        if _cmp['only_in_hevy']:
                            st.write("**Solo in Hevy**")
                            for _nm in _cmp['only_in_hevy']:
                                st.write(f"  • {_nm}")
                        if _cmp['only_in_app']:
                            st.write("**Solo nell'app**")
                            for _nm in _cmp['only_in_app']:
                                st.write(f"  • {_nm}")
                        if _cmp['load_diffs']:
                            st.write("**Carichi diversi**")
                            st.dataframe(
                                pd.DataFrame([
                                    {
                                        'Esercizio': d['exercise'],
                                        'App':       d['app_value'],
                                        'Hevy':      d['hevy_value'],
                                    }
                                    for d in _cmp['load_diffs']
                                ]),
                                hide_index=True,
                            )

                # ── Step 4: import button ──────────────────────────────────
                if _Y > 0:
                    if st.button(
                        f"Importa {_Y} sessioni nuove",
                        type="primary",
                        key="hevy_import_btn",
                    ):
                        _rows_df  = sessions_to_csv_rows(_new_sess, load_exercise_map())
                        _n_written = import_hevy_rows(_rows_df)
                        st.success(
                            f"Importate {_n_written} righe. "
                            "Ricarica l'app per aggiornare i grafici."
                        )
                elif _Z > 0:
                    st.info("Tutte le sessioni sono già presenti.")

            except Exception as _hevy_err:
                st.error(f"Errore durante il parsing del CSV: {_hevy_err}")