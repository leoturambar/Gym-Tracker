import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from config import DAYS, MUSCLES
from data_manager import (
    init_files, load_sessions, save_session, delete_session,
    get_last_values, save_bodyweight, load_bodyweight,
    get_bodyweight_on, load_memory, save_memory
)
from metrics import (
    compute_muscle_scores, normalize_scores,
    enrich_with_bodyweight, filter_by_period,
    exercise_progression
)
from llm import get_llm_analysis

# ── Setup ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gym Tracker",
    page_icon="🏋️",
    layout="wide"
)

init_files()

# ── Tab structure ─────────────────────────────────────────────────────────────

tab_log, tab_storico, tab_progressi, tab_radar, tab_llm, tab_profilo = st.tabs([
    "📋 Log", "📅 Storico", "📈 Progressi", "🕸️ Radar", "🤖 Analisi AI", "👤 Profilo"
])

with tab_log:
    st.header("Registra sessione")

    # ── Selezione giorno ──────────────────────────────────────────────────
    day_options = {f"Day {d['id']} — {d['name']} ({d['sub']})": d for d in DAYS}
    selected_label = st.selectbox("Giorno", list(day_options.keys()))
    selected_day = day_options[selected_label]

    # ── Data ──────────────────────────────────────────────────────────────
    session_date = st.date_input("Data", value=pd.Timestamp.today())
    session_date_str = str(session_date)

    # ── Pre-popola con ultimi valori ──────────────────────────────────────
    last_values = get_last_values(selected_day['id'])

    st.divider()

    # ── Esercizi ──────────────────────────────────────────────────────────
    st.subheader("Esercizi")

    exercise_inputs = {}
    for ex in selected_day['exercises']:
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.write(f"**{ex['name']}**")

        with col2:
            if ex['type'] == 'excluded':
                st.caption("attivazione — non contato")
                exercise_inputs[ex['name']] = {'value': 0, 'skipped': False}
                continue
            elif ex['type'] == 'bodyweight':
                st.caption("corpo libero")
                default_val = 0
            elif ex['type'] == 'timed':
                default_val = last_values.get(ex['name'], ex.get('default', 60))
            else:
                default_val = last_values.get(ex['name'], ex.get('default', 0))

            label = "secondi" if ex['type'] == 'timed' else "kg"

            if ex['type'] == 'bodyweight':
                exercise_inputs[ex['name']] = {'value': 0, 'skipped': False}
            else:
                value = st.number_input(
                    label,
                    value=float(default_val),
                    step=0.5 if ex['type'] != 'timed' else 5.0,
                    key=f"val_{ex['name']}",
                    label_visibility="collapsed"
                )
                exercise_inputs[ex['name']] = {'value': value, 'skipped': False}

        with col3:
            skipped = st.checkbox("salta", key=f"skip_{ex['name']}")
            if ex['name'] in exercise_inputs:
                exercise_inputs[ex['name']]['skipped'] = skipped

    st.divider()

    # ── Note ──────────────────────────────────────────────────────────────
    note = st.text_area("Note sessione", placeholder="Come ti sei sentito, variazioni, dolori…")

    # ── Salva ─────────────────────────────────────────────────────────────
    if st.button("💾 Salva sessione", type="primary"):
        session_id = int(time.time())
        exercises = []
        for ex in selected_day['exercises']:
            inp = exercise_inputs.get(ex['name'], {'value': 0, 'skipped': False})
            exercises.append({
                'name':    ex['name'],
                'type':    ex['type'],
                'value':   inp['value'],
                'skipped': inp['skipped'],
            })
        save_session(session_id, session_date_str,
                     selected_day['id'], selected_day['name'],
                     exercises, note)
        st.success("Sessione salvata!")
        st.rerun()

with tab_storico:
    st.header("Storico sessioni")

    df = load_sessions()

    if df.empty:
        st.info("Nessuna sessione registrata ancora.")
    else:
        # lista sessioni uniche ordinate per data decrescente
        sessions_meta = (
            df[['session_id', 'date', 'day_name', 'note']]
            .drop_duplicates(subset='session_id')
            .sort_values('date', ascending=False)
        )

        for _, meta in sessions_meta.iterrows():
            sid = meta['session_id']
            date = meta['date']
            day_name = meta['day_name']
            note = meta['note'] if pd.notna(meta['note']) else ''

            with st.expander(f"**{date}** — {day_name}"):
                # esercizi della sessione
                sess_df = df[df['session_id'] == sid]
                for _, row in sess_df.iterrows():
                    if row['skipped']:
                        st.write(f"~~{row['exercise']}~~ — saltato")
                    elif row['type'] == 'excluded':
                        continue
                    elif row['type'] == 'timed':
                        st.write(f"**{row['exercise']}**: {int(row['value'])}s")
                    elif row['type'] == 'bodyweight':
                        st.write(f"**{row['exercise']}**: corpo libero")
                    else:
                        st.write(f"**{row['exercise']}**: {row['value']} kg")

                if note:
                    st.caption(f"📝 {note}")

                # bottone elimina
                if st.button("🗑️ Elimina sessione", key=f"del_{sid}"):
                    delete_session(sid)
                    st.rerun()

with tab_progressi:
    st.header("Progressione esercizi")

    # lista esercizi con peso (escludi bodyweight, timed, excluded)
    weighted_exercises = [
        ex['name']
        for day in DAYS
        for ex in day['exercises']
        if ex['type'] in ('weighted', 'weighted_bw')
    ]
    weighted_exercises = sorted(set(weighted_exercises))

    selected_ex = st.selectbox("Esercizio", weighted_exercises)

    df_prog = exercise_progression(selected_ex)

    if df_prog.empty:
        st.info("Nessun dato per questo esercizio ancora.")
    else:
        # metriche riassuntive
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sessioni", len(df_prog))
        with col2:
            st.metric("Massimo", f"{df_prog['value'].max():.1f} kg")
        with col3:
            st.metric("Attuale", f"{df_prog['value'].iloc[-1]:.1f} kg")
        with col4:
            delta = df_prog['value'].iloc[-1] - df_prog['value'].iloc[0]
            st.metric("Progresso", f"{delta:+.1f} kg")

        st.divider()

    # grafico carico nel tempo
    st.subheader("Carico (kg)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prog['date'],
        y=df_prog['value'],
        mode='lines+markers',
        name='kg',
        line=dict(color='#00c878', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        yaxis=dict(
            range=[
                df_prog['value'].min() * 0.85,
                df_prog['value'].max() * 1.15
            ]
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # grafico RTV nel tempo
    if df_prog['rtv'].sum() > 0:
        st.subheader("RTV nel tempo")
        fig_rtv = go.Figure()
        fig_rtv.add_trace(go.Scatter(
            x=df_prog['date'],
            y=df_prog['rtv'],
            mode='lines+markers',
            name='RTV',
            line=dict(color='#5090ff', width=2),
            marker=dict(size=8)
        ))
        fig_rtv.update_layout(
            yaxis=dict(
                range=[
                    df_prog['rtv'].min() * 0.85,
                    df_prog['rtv'].max() * 1.15
                ]
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
        )
        st.plotly_chart(fig_rtv, use_container_width=True)
    else:
        st.caption("Imposta il peso corporeo nel tab Profilo per vedere il grafico RTV.")

with tab_radar:
    st.header("Radar muscolare")

    df_all = load_sessions()

    if df_all.empty:
        st.info("Nessuna sessione registrata ancora.")
    else:
        df_all = enrich_with_bodyweight(df_all)

        col1, col2 = st.columns(2)
        with col1:
            metric = st.radio(
                "Metrica",
                options=['freq', 'rtv'],
                format_func=lambda x: 'Frequenza esercizi' if x == 'freq' else 'Carico relativo (RTV)',
                horizontal=True
            )
        with col2:
            period = st.radio(
                "Confronta",
                options=['all', 'week', 'month', 'year'],
                format_func=lambda x: {
                    'all':   'vs Schema',
                    'week':  'vs Sett. prec.',
                    'month': 'vs Mese prec.',
                    'year':  'vs Anno prec.'
                }[x],
                horizontal=True
            )

        # avviso se RTV ma nessun peso corporeo
        if metric == 'rtv' and df_all['bodyweight'].isna().all():
            st.warning("Imposta il peso corporeo nel tab Profilo per usare la metrica RTV.")

        # calcolo scores
        if period == 'all':
            scores_a = normalize_scores(compute_muscle_scores(df_all, metric))
            # schema come riferimento
            from config import EX_MUSCLES
            schema_scores = {m: 0.0 for m in MUSCLES}
            for day in DAYS:
                for ex in day['exercises']:
                    for m in EX_MUSCLES.get(ex['name'], []):
                        schema_scores[m] += 1.0
            max_s = max(schema_scores.values()) or 1
            scores_b = {m: v / max_s for m, v in schema_scores.items()}
            label_a, label_b = 'Reale', 'Schema pianificato'
        else:
            df_cur, df_prev = filter_by_period(df_all, period)
            scores_a = normalize_scores(compute_muscle_scores(df_cur, metric))
            scores_b = normalize_scores(compute_muscle_scores(df_prev, metric))
            label_a, label_b = {
                'week':  ('Questa settimana', 'Settimana scorsa'),
                'month': ('Questo mese',      'Mese scorso'),
                'year':  ("Quest'anno",        'Anno scorso'),
            }[period]

        # radar chart con plotly
        categories = MUSCLES + [MUSCLES[0]]
        values_a = [scores_a.get(m, 0) for m in MUSCLES] + [scores_a.get(MUSCLES[0], 0)]
        values_b = [scores_b.get(m, 0) for m in MUSCLES] + [scores_b.get(MUSCLES[0], 0)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_b, theta=categories,
            fill='toself', name=label_b,
            line=dict(color='#5090ff', dash='dash', width=1.5),
            fillcolor='rgba(80,144,255,0.1)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=values_a, theta=categories,
            fill='toself', name=label_a,
            line=dict(color='#00c878', width=2),
            fillcolor='rgba(0,200,120,0.18)'
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(240,240,240,0.1)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
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
        
        st.plotly_chart(fig, use_container_width=True)

        # barre orizzontali
        st.subheader(f"{label_a} — {'frequenza' if metric == 'freq' else 'RTV'}")
        for m in MUSCLES:
            v = scores_a.get(m, 0)
            st.progress(v, text=f"{m}: {v:.2f}")

with tab_llm:
    st.header("Analisi AI")

    df_all = load_sessions()

    if df_all.empty:
        st.info("Logga almeno una sessione per usare l'analisi AI.")
    else:
        df_all = enrich_with_bodyweight(df_all)

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
            df_cur = df_all
            df_prev = None
        else:
            df_cur, df_prev = filter_by_period(df_all, period_llm)

        # profilo utente
        current_bw = get_bodyweight_on(str(pd.Timestamp.today().date()))
        user_profile = {
            'bodyweight': current_bw,
            'goal':       st.session_state.get('goal', 'non specificato'),
            'memory':     load_memory(),
        }

        st.divider()

        if st.button("🤖 Genera analisi", type="primary"):
            with st.spinner("Patches O'Houlihan sta analizzando i tuoi allenamenti…"):
                try:
                    analysis = get_llm_analysis(user_profile, df_cur, df_prev, focus)
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
        bw_value = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, step=0.5, key="bw_value")
        if st.button("💾 Salva peso", type="primary"):
            save_bodyweight(bw_value, str(bw_date))
            st.success(f"Peso {bw_value} kg salvato per il {bw_date}.")
            st.rerun()

    with col2:
        if not df_bw.empty:
            st.caption("Storico peso corporeo")
            st.dataframe(
                df_bw.sort_values('date', ascending=False).reset_index(drop=True),
                use_container_width=True,
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
    goal = st.selectbox("Obiettivo principale", [
        "Ipertrofia (aumento massa muscolare)",
        "Forza (aumento carichi)",
        "Resistenza muscolare",
        "Ricomposizione corporea",
        "Mantenimento",
    ])
    st.session_state['goal'] = goal