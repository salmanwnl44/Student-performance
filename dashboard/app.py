import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="EduPulse — Student Analytics", layout="wide", page_icon="◉", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    [data-testid="stToolbar"] { display: none !important; }
    header { background: transparent !important; }

    /* Lock sidebar open — hide every collapse/expand control */
    [data-testid="collapsedControl"]  { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    button[kind="header"]             { display: none !important; }
    .st-emotion-cache-zq5wmm          { display: none !important; }
    section[data-testid="stSidebar"] > div:first-child > button { display: none !important; }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp { background: #12151c; font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: #181c25;
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    h1, h2, h3, h4 { color: #e8e6e3 !important; font-weight: 600; }

    .card {
        background: #1a1e28;
        border: 1px solid #24293a;
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 14px;
        transition: border-color 0.2s ease;
    }
    .card:hover { border-color: #3a3f52; }
    .card-accent-amber { border-left: 3px solid #d4a053; }
    .card-accent-coral { border-left: 3px solid #c75c5c; }
    .card-accent-sage  { border-left: 3px solid #6b9e7a; }
    .card-accent-blue  { border-left: 3px solid #5b7fb5; }

    .t-overline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .t-display  { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; margin: 0; }
    .t-body     { color: #9ca3af; font-size: 0.88rem; line-height: 1.55; }
    .t-caption  { color: #6b7280; font-size: 0.78rem; }
    .t-mono     { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #d4a053; }

    .c-amber { color: #d4a053; }
    .c-coral { color: #c75c5c; }
    .c-sage  { color: #6b9e7a; }
    .c-blue  { color: #5b7fb5; }
    .c-ghost { color: #4b5563; }
    .c-light { color: #e8e6e3; }

    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; }
    .pill-danger { background: #2a1a1a; color: #c75c5c; border: 1px solid #3d2424; }
    .pill-safe   { background: #1a2a1e; color: #6b9e7a; border: 1px solid #243d28; }
    .pill-warn   { background: #2a2618; color: #d4a053; border: 1px solid #3d3522; }

    .divider { height: 1px; background: linear-gradient(90deg, transparent, #24293a 20%, #24293a 80%, transparent); margin: 16px 0; }

    .sidebar-section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        font-weight: 600;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin: 14px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .stButton>button {
        background: #d4a053;
        color: #12151c;
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 10px 28px;
    }
    .stButton>button:hover { background: #e0b06a; box-shadow: 0 4px 16px rgba(212,160,83,0.25); }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #24293a; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px 6px 0 0; padding: 10px 22px; color: #6b7280; font-weight: 500; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { background: #1a1e28 !important; color: #d4a053 !important; border-bottom: 2px solid #d4a053; }

    [data-testid="stMetric"] { background: #1a1e28; border: 1px solid #24293a; border-radius: 8px; padding: 14px; }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #e8e6e3; }

    .stForm { border: none !important; padding: 0 !important; }

    /* Tighten sidebar widget spacing */
    section[data-testid="stSidebar"] .stSlider { margin-bottom: 4px; }
    section[data-testid="stSidebar"] .stNumberInput { margin-bottom: 4px; }
    section[data-testid="stSidebar"] .stSelectbox { margin-bottom: 4px; }
    section[data-testid="stSidebar"] .stCheckbox { margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

ICON = {
    "shield":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "alert":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "check":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "book":      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
    "clock":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "target":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "users":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>',
    "trending":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "award":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "briefcase": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/></svg>',
}

if "current_dropout_prob" not in st.session_state:
    st.session_state.current_dropout_prob = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── HEADER ──
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:8px;">
  <div style="width:40px; height:40px; background:linear-gradient(135deg,#d4a053,#c7893e); border-radius:8px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
    <span style="font-size:1.2rem; font-weight:900; color:#12151c;">E</span>
  </div>
  <div>
    <h1 style="margin:0; font-size:1.6rem; letter-spacing:-0.5px;">EduPulse AI Mentor</h1>
    <p class="t-caption" style="margin:0;">Student analytics &amp; intervention engine</p>
  </div>
</div>
    """, unsafe_allow_html=True)

with col2:
    if st.session_state.current_dropout_prob is not None:
        p = st.session_state.current_dropout_prob
        c = "c-coral" if p > 0.5 else "c-amber" if p > 0.2 else "c-sage"
        i = "High" if p > 0.5 else "Medium" if p > 0.2 else "Low"
        st.markdown(
            f"<div style='text-align:right; padding-top:5px;'>"
            f"<p class='t-overline'>Dropout Risk</p>"
            f"<h2 class='t-display {c}' style='margin:0;'>{p:.1%}</h2>"
            f"<p class='t-caption {c}'>{i} Index</p></div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    dev_mode = st.toggle("Developer Mode", value=False)

    _cb1, _cb2 = st.columns(2)
    with _cb1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_dropout_prob = None
            if "artifact_data" in st.session_state:
                del st.session_state.artifact_data
            st.rerun()
    with _cb2:
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast("Cache cleared.", icon="✓")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<p class='t-overline'>student profile</p>", unsafe_allow_html=True)

    with st.form("student_form"):

        # ── Demographics ──
        st.markdown(
            f"<p class='sidebar-section-label'>{ICON['users']} Demographics</p>",
            unsafe_allow_html=True
        )
        age    = st.number_input("Age", 15, 100, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
        dept   = st.selectbox("Department", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Medicine"])
        sem    = st.slider("Semester", 1, 8, 3)
        pedu   = st.selectbox("Parent Education", ["No Formal Education", "High School", "Bachelor", "Master", "PhD"])
        finc   = st.selectbox("Family Income", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: schol = st.checkbox("Scholarship")
        with col_s2: inet  = st.checkbox("Internet", value=True)
        with col_s3: job   = st.checkbox("Part-Time Job")

        # ── Academics ──
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='sidebar-section-label'>{ICON['book']} Academics</p>",
            unsafe_allow_html=True
        )
        gpa  = st.slider("GPA", 0.0, 4.0, 2.8, 0.1)
        att  = st.slider("Attendance %", 0.0, 100.0, 78.0)
        mid  = st.slider("Midterm Score", 0.0, 100.0, 65.0)
        asgn = st.slider("Assignment Score", 0.0, 100.0, 70.0)
        miss = st.number_input("Missed Deadlines", 0, 50, 2)

        # ── Engagement ──
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='sidebar-section-label'>{ICON['trending']} Engagement</p>",
            unsafe_allow_html=True
        )
        commute = st.slider("Commute (miles)", 0.0, 80.0, 10.0)
        tmat    = st.slider("LMS Hours / Week", 0.0, 60.0, 10.0)
        study   = st.slider("Study Hours / Week", 0.0, 40.0, 8.0)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        go_btn = st.form_submit_button("Run Analysis", use_container_width=True)

# Removed inputs kept at sensible defaults for API payload
payload = {
    "age": age,
    "gender": gender,
    "department": dept,
    "semester": sem,
    "parent_education": pedu,
    "family_income": finc,
    "commute_distance": commute,
    "scholarship_status": int(schol),
    "has_internet_access": int(inet),
    "part_time_job": int(job),
    "gpa_current": gpa,
    "attendance_rate": att,
    "midterm_score": mid,
    "assignment_score": asgn,
    "quiz_score": 60.0,           # removed from UI — sensible default
    "login_frequency": 8,         # removed from UI — sensible default
    "forum_participation_score": 3,  # removed from UI — sensible default
    "time_spent_on_materials": tmat,
    "study_hours_weekly": study,
    "library_usage_weekly": 2,    # removed from UI — sensible default
    "missed_deadlines_count": miss,
}


def render_risk_widgets(d, widget_key=""):
    risk = d["student_risk_assessment"]
    prob = d["dropout_probability"]

    pill_cls = "pill-danger" if risk == "High" else "pill-warn" if risk == "Medium" else "pill-safe"
    icon = ICON["alert"] if risk == "High" else ICON["check"]
    pc  = "c-coral" if prob > 0.5 else "c-amber" if prob > 0.2 else "c-sage"
    gc  = "c-coral" if gpa  < 2.0  else "c-amber" if gpa  < 3.0  else "c-sage"
    ac  = "c-coral" if att  < 70   else "c-amber" if att  < 85   else "c-sage"
    mc2 = "c-coral" if mid  < 50   else "c-amber" if mid  < 70   else "c-sage"

    st.markdown(f"""
    <div style='display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px;'>
        <div class='card' style='flex:1; min-width:110px; margin-bottom:0;'>
            <p class='t-overline'>status</p>
            <span class='pill {pill_cls}'>{icon} {risk} Risk</span>
        </div>
        <div class='card' style='flex:1; min-width:90px; margin-bottom:0;'>
            <p class='t-overline'>probability</p>
            <p class='t-display {pc}'>{prob:.1%}</p>
        </div>
        <div class='card' style='flex:1; min-width:70px; margin-bottom:0;'>
            <p class='t-overline'>gpa</p>
            <p class='t-display {gc}'>{gpa:.2f}</p>
        </div>
        <div class='card' style='flex:1; min-width:70px; margin-bottom:0;'>
            <p class='t-overline'>attendance</p>
            <p class='t-display {ac}'>{att:.0f}%</p>
        </div>
        <div class='card' style='flex:1; min-width:70px; margin-bottom:0;'>
            <p class='t-overline'>midterm</p>
            <p class='t-display {mc2}'>{mid:.0f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cr, cg = st.columns([3, 2])

    with cr:
        with st.container(border=True):
            st.markdown("<p class='t-overline'>performance profile</p>", unsafe_allow_html=True)
            cats = ['Attendance', 'GPA', 'Midterm', 'Study', 'LMS Hrs', 'Deadlines']
            sv   = [att / 10, (gpa / 4) * 10, mid / 10, min(study / 2, 10), min(tmat / 6, 10), max(10 - miss * 1.5, 0)]
            bv   = [8.5, 7.5, 7, 7, 7, 9]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=bv, theta=cats, fill='toself', name='Cohort Avg',
                line=dict(color='#3a3f52', width=1), fillcolor='rgba(58,63,82,0.15)'))
            fig.add_trace(go.Scatterpolar(r=sv, theta=cats, fill='toself', name='This Student',
                line=dict(color='#d4a053', width=2), fillcolor='rgba(212,160,83,0.12)'))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], color="#2a2e3a", gridcolor="#24293a"),
                    angularaxis=dict(color="#6b7280", gridcolor="#24293a"),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                legend=dict(font=dict(color='#6b7280', size=11)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#6b7280', family='Inter'),
                height=370,
                margin=dict(t=50, b=30, l=80, r=80)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"radar_{widget_key}")

    with cg:
        with st.container(border=True):
            st.markdown("<p class='t-overline'>risk meter</p>", unsafe_allow_html=True)
            bar_col = "#c75c5c" if prob > 0.5 else "#d4a053" if prob > 0.2 else "#6b9e7a"
            fg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number=dict(suffix="%", font=dict(color='#e8e6e3', size=32, family='JetBrains Mono')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor='#2a2e3a', tickfont=dict(color='#4b5563')),
                    bar=dict(color=bar_col, thickness=0.7),
                    bgcolor='#1a1e28',
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 20],  color='#1a2a1e'),
                        dict(range=[20, 50], color='#2a2618'),
                        dict(range=[50, 100], color='#2a1a1a'),
                    ]
                )
            ))
            fg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#6b7280'),
                height=300,
                margin=dict(t=50, b=20, l=30, r=30)
            )
            st.plotly_chart(fg, use_container_width=True, key=f"gauge_{widget_key}")


@st.fragment
def render_whatif_simulator(data, att, study, payload):
    st.markdown("#### What-If Simulator")
    st.markdown("<p class='t-caption'>Tweak factors to see how dropout probability changes.</p>", unsafe_allow_html=True)
    with st.container(border=True):
        sim_att   = st.slider("Simulated Attendance (%)", 0, 100, int(att), key="sim_att")
        sim_study = st.slider("Simulated Study Hrs/Wk",   0, 40,  int(study), key="sim_st")

        if st.button("Recalculate Risk", key="sim_btn"):
            p2 = payload.copy()
            p2["attendance_rate"]    = sim_att
            p2["study_hours_weekly"] = sim_study
            with st.spinner("Simulating..."):
                r2 = requests.post(f"{API_BASE}/predict/dropout", json=p2)
                if r2.status_code == 200:
                    d2       = r2.json()
                    new_risk = d2["dropout_probability"]
                    delta    = data["dropout_probability"] - new_risk
                    if delta > 0:
                        st.markdown(f"**New Risk:** <span class='t-display c-sage'>{new_risk:.1%}</span> — reduced by {delta:.1%}", unsafe_allow_html=True)
                    elif delta < 0:
                        st.markdown(f"**New Risk:** <span class='t-display c-coral'>{new_risk:.1%}</span> — increased by {-delta:.1%}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**New Risk:** <span class='t-display c-amber'>{new_risk:.1%}</span> — no change", unsafe_allow_html=True)


def render_artifact_panel(data, gpa, att, mid, study, miss, payload):
    st.markdown("<h3 style='margin-bottom:20px; font-size:1.3rem'>Analysis Report</h3>", unsafe_allow_html=True)

    st.markdown("#### Explainable AI Reasoning")
    st.info(data.get("reasoning", "No reasoning provided."))

    st.markdown("#### Recommended Action Plans")
    if data.get("intervention"):
        for m in data["intervention"].split("\n"):
            if m.strip():
                st.markdown(m)

    st.markdown("<hr/>", unsafe_allow_html=True)

    if data.get("quantified_plans"):
        for p in data["quantified_plans"]:
            st.success(f"**{p['action']}**  \n*Estimated Impact: Reduces risk by {p['reduction']:.1%}*")

    st.markdown("#### Risk Trends & Influencers")
    c1, c2 = st.columns(2)

    with c1:
        if "risk_history" in data:
            df_t = pd.DataFrame(data["risk_history"])
            ft = px.line(df_t, x="week", y="risk", title="5-Week Risk Trend",
                         markers=True, color_discrete_sequence=['#c75c5c'])
            ft.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#6b7280'), height=250,
                              margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(ft, use_container_width=True, key="artifact_risk_trend")

    with c2:
        if "feature_importance" in data:
            df_f = pd.DataFrame(list(data["feature_importance"].items()), columns=["Feature", "Impact"])
            ff = px.bar(df_f, x="Impact", y="Feature", orientation='h',
                        title="Top Risk Drivers", color_discrete_sequence=['#d4a053'])
            ff.update_layout(yaxis={'categoryorder': 'total ascending'},
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#6b7280'), height=250,
                             margin=dict(t=30, b=10, l=150, r=10))
            st.plotly_chart(ff, use_container_width=True, key="artifact_feature_importance")

    render_whatif_simulator(data, att, study, payload)


def render_main_assistant():
    col_chat = st.container()

    with col_chat:
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg.get("type") == "risk_report":
                    st.markdown("**Analysis complete.** Here is the interactive risk report:")
                    render_risk_widgets(msg["data"], widget_key=msg.get("id", str(msg_idx)))
                    st.markdown(msg["content"])
                    if not any(m.get("type") == "artifact_panel" for m in st.session_state.messages):
                        st.markdown(
                            "<br/>I can generate a detailed action plan with AI reasoning, trends, and simulations.",
                            unsafe_allow_html=True
                        )
                        if st.button("Create Detailed Report", key=f"btn_gen_{msg.get('id', str(msg_idx))}", type="primary"):
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "artifact_panel",
                                "content": "I've drafted the detailed plan:",
                            })
                            st.rerun()

                elif msg.get("type") == "artifact_panel":
                    st.markdown(msg["content"])
                    if "artifact_data" in st.session_state and st.session_state.artifact_data:
                        if "reasoning" not in st.session_state.artifact_data:
                            with st.spinner("Drafting detailed plan..."):
                                r_llm = requests.post(f"{API_BASE}/predict/analyze", json=payload)
                                if r_llm.status_code == 200:
                                    llm_data = r_llm.json()
                                    st.session_state.artifact_data["reasoning"]     = llm_data["reasoning"]
                                    st.session_state.artifact_data["intervention"]  = llm_data["intervention"]
                                    st.rerun()
                        with st.container(border=True):
                            render_artifact_panel(
                                st.session_state.artifact_data,
                                gpa, att, mid, study, miss, payload
                            )
                else:
                    st.markdown(msg["content"])

        if go_btn:
            import uuid
            loading_slot = st.empty()
            loading_slot.markdown("""
<div style="
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding: 60px 20px; background:#1a1e28; border:1px solid #24293a;
    border-radius:12px; margin:20px 0; text-align:center;">
  <div style="
      width:48px; height:48px; border:3px solid #24293a;
      border-top:3px solid #d4a053; border-radius:50%;
      animation:spin 0.9s linear infinite; margin-bottom:20px;">
  </div>
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
     color:#d4a053; letter-spacing:2px; text-transform:uppercase; margin:0 0 6px 0;">
    Analysing Student Profile
  </p>
  <p style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#4b5563; margin:0;">
    Running ML model &amp; generating risk assessment…
  </p>
</div>
<style>
@keyframes spin { to { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)
            r = requests.post(f"{API_BASE}/predict/dropout", json=payload)
            loading_slot.empty()
            if r.status_code == 200:
                d = r.json()
                d["id"] = str(uuid.uuid4())
                st.session_state.current_dropout_prob = d["dropout_probability"]
                if "artifact_data" not in st.session_state:
                    st.session_state.artifact_data = {}
                st.session_state.artifact_data.update(d)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "risk_report",
                    "data": d,
                    "content": "*Student profile analysed.*",
                    "id": d["id"],
                })
                st.rerun()

    if prompt := st.chat_input("Ask how the student can improve..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with col_chat:
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    response = requests.post(f"{API_BASE}/chat/student", json={"student": payload, "message": prompt})
                    response.raise_for_status()
                    answer = response.json().get("response", "No answer found.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    if "plan" in prompt.lower() and not any(m.get("type") == "artifact_panel" for m in st.session_state.messages):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "artifact_panel",
                            "content": "I've drafted the detailed plan:",
                        })
                        st.rerun()
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}")


def render_model_comparison():
    try:
        mr = requests.get(f"{API_BASE}/metrics")
        if mr.status_code == 200:
            md   = mr.json()
            best = md.get("best_model", "rf")
            ds   = md.get("dataset_size", "?")
            fc   = md.get("features_count", "?")

            st.markdown("<p class='t-overline'>training summary</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='card'><p class='t-body'>{ICON['award']} Champion: "
                f"<strong class='c-amber'>{md.get(best, {}).get('name', best)}</strong>"
                f" &nbsp;·&nbsp; <span class='t-mono'>{ds:,}</span> samples"
                f" &nbsp;·&nbsp; <span class='t-mono'>{fc}</span> features</p></div>",
                unsafe_allow_html=True
            )

            rows = []
            for k in ['rf', 'xgb', 'lgbm']:
                if k in md:
                    m = md[k]['metrics']
                    rows.append({
                        'Model':     md[k]['name'],
                        'Accuracy':  f"{m['accuracy']:.2%}",
                        'Precision': f"{m['precision']:.2%}",
                        'Recall':    f"{m['recall']:.2%}",
                        'F1':        f"{m['f1_score']:.2%}",
                        'AUC':       f"{m['roc_auc']:.4f}" if isinstance(m['roc_auc'], float) else m['roc_auc'],
                        ' ':         '◆' if k == best else '',
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            bd = []
            for k in ['rf', 'xgb', 'lgbm']:
                if k in md:
                    m, n = md[k]['metrics'], md[k]['name']
                    for mn in ['accuracy', 'precision', 'recall', 'f1_score']:
                        bd.append({'Model': n, 'Metric': mn.replace('_', ' ').title(), 'Value': m[mn]})
            if bd:
                fb = px.bar(pd.DataFrame(bd), x='Metric', y='Value', color='Model', barmode='group',
                            color_discrete_sequence=['#5b7fb5', '#d4a053', '#6b9e7a'])
                fb.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#6b7280', family='Inter'),
                    yaxis=dict(range=[0.9, 1.0], gridcolor='#1e2230', title=''),
                    xaxis=dict(gridcolor='#1e2230', title=''),
                    height=380,
                    legend=dict(font=dict(color='#9ca3af'))
                )
                st.plotly_chart(fb, use_container_width=True, key="model_comparison_bar")
        else:
            st.info("No training data available yet.")
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")


def render_data_explorer():
    try:
        df = pd.read_csv("data/raw/synthetic_student_data.csv")

        st.markdown("<p class='t-overline'>dataset overview</p>", unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: st.metric("Records",      f"{len(df):,}")
        with s2: st.metric("Avg GPA",      f"{df['gpa_current'].mean():.2f}")
        with s3: st.metric("Attendance",   f"{df['attendance_rate'].mean():.1f}%")
        with s4: st.metric("Dropout Rate", f"{df['dropout_risk'].mean():.1%}")
        with s5: st.metric("Missing Cells",f"{df.isnull().sum().sum():,}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<p class='t-overline' style='margin-top:16px'>grade distribution</p>", unsafe_allow_html=True)
            gc = df['final_grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'])
            fg = px.bar(x=gc.index, y=gc.values, labels={'x': '', 'y': ''}, color_discrete_sequence=['#d4a053'])
            fg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#6b7280', family='Inter'), showlegend=False, height=300,
                             xaxis=dict(gridcolor='#1e2230'), yaxis=dict(gridcolor='#1e2230'))
            st.plotly_chart(fg, use_container_width=True, key="explorer_grade_dist")

        with c2:
            st.markdown("<p class='t-overline' style='margin-top:16px'>dropout rate by department</p>", unsafe_allow_html=True)
            dr = df.groupby('department')['dropout_risk'].mean().sort_values(ascending=True)
            fd = px.bar(x=dr.values, y=dr.index, labels={'x': '', 'y': ''},
                        orientation='h', color_discrete_sequence=['#c75c5c'])
            fd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#6b7280', family='Inter'), showlegend=False, height=300,
                             xaxis=dict(gridcolor='#1e2230', tickformat='.0%'), yaxis=dict(gridcolor='#1e2230'))
            st.plotly_chart(fd, use_container_width=True, key="explorer_dropout_dept")

        st.markdown("<p class='t-overline' style='margin-top:16px'>feature correlation matrix</p>", unsafe_allow_html=True)
        ndf  = df.select_dtypes(include=[np.number])
        corr = ndf.corr()
        fcc = px.imshow(corr, text_auto='.2f',
                        color_continuous_scale=[[0, '#1a1e28'], [0.5, '#2a2e3a'], [1, '#d4a053']],
                        zmin=-1, zmax=1, aspect='auto')
        fcc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280', family='Inter', size=8),
                          height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fcc, use_container_width=True, key="explorer_corr_matrix")

        st.markdown("<p class='t-overline' style='margin-top:16px'>sample records</p>", unsafe_allow_html=True)
        st.dataframe(df.head(80), use_container_width=True, height=350)

    except FileNotFoundError:
        st.info("No dataset found at data/raw/synthetic_student_data.csv.")


# ── MAIN RENDER ──
if dev_mode:
    t1, t2, t3 = st.tabs(["AI Mentor", "Model Comparison", "Data Explorer"])
    with t1: render_main_assistant()
    with t2: render_model_comparison()
    with t3: render_data_explorer()
else:
    render_main_assistant()
