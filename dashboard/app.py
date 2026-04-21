import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

API_BASE = "http://localhost:8000"


# ── Resilient API helpers (retry on connection reset) ──────────────────────
def _api_request(method: str, url: str, **kwargs):
    kwargs.setdefault("timeout", 30)
    for attempt in range(2):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 0:
                time.sleep(0.6)
    return None

def api_post(path: str, payload: dict):
    return _api_request("POST", f"{API_BASE}{path}", json=payload)

def api_get(path: str):
    return _api_request("GET", f"{API_BASE}{path}")

st.set_page_config(page_title="EduPulse — Student Analytics", layout="wide", page_icon="◉", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Lock & hide sidebar controls ── */
[data-testid="collapsedControl"]  { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[kind="header"]             { display: none !important; }
.st-emotion-cache-zq5wmm          { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > button { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }
header                            { background: transparent !important; }

/* ── Base ── */
.stApp {
    background: #0d1017;
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1017; }
::-webkit-scrollbar-thumb { background: #2a2f3d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #d4a053; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161f 0%, #111420 100%);
    border-right: 1px solid rgba(212,160,83,0.08);
    box-shadow: 4px 0 24px rgba(0,0,0,0.4);
}
section[data-testid="stSidebar"] .stSlider     { margin-bottom: 2px; }
section[data-testid="stSidebar"] .stNumberInput{ margin-bottom: 2px; }
section[data-testid="stSidebar"] .stSelectbox  { margin-bottom: 2px; }
section[data-testid="stSidebar"] .stCheckbox   { margin-bottom: 0px; }

/* Slider thumb amber */
[data-testid="stSlider"] [role="slider"] {
    background: #d4a053 !important;
    box-shadow: 0 0 0 3px rgba(212,160,83,0.18) !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:first-child {
    background: #d4a053 !important;
}

/* Inputs */
[data-baseweb="input"] input,
[data-baseweb="select"] div,
[data-baseweb="textarea"] textarea {
    background: #181c28 !important;
    border-color: #252a3a !important;
    color: #e0ddd9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="input"] input:focus,
[data-baseweb="select"] [aria-selected="true"] {
    border-color: #d4a053 !important;
    box-shadow: 0 0 0 2px rgba(212,160,83,0.12) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #8a8fa8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Typography ── */
h1, h2, h3, h4 { color: #edeae6 !important; font-weight: 700; letter-spacing: -0.3px; }

.t-overline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    color: #525868;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 6px;
}
.t-display { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1; margin: 0; }
.t-body    { color: #9198ab; font-size: 0.88rem; line-height: 1.6; }
.t-caption { color: #525868; font-size: 0.78rem; }
.t-mono    { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #d4a053; }

/* ── Colors ── */
.c-amber { color: #d4a053; }
.c-coral { color: #c75c5c; }
.c-sage  { color: #6b9e7a; }
.c-light { color: #edeae6; }

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #161b26 0%, #141820 100%);
    border: 1px solid #1e2436;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.15s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.card:hover {
    border-color: rgba(212,160,83,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(212,160,83,0.06);
    transform: translateY(-1px);
}

/* ── Status Pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.5px;
}
.pill-danger {
    background: rgba(199,92,92,0.12); color: #e07070;
    border: 1px solid rgba(199,92,92,0.25);
    box-shadow: 0 0 12px rgba(199,92,92,0.08);
}
.pill-safe {
    background: rgba(107,158,122,0.12); color: #7dbc92;
    border: 1px solid rgba(107,158,122,0.25);
    box-shadow: 0 0 12px rgba(107,158,122,0.08);
}
.pill-warn {
    background: rgba(212,160,83,0.12); color: #d4a053;
    border: 1px solid rgba(212,160,83,0.25);
    box-shadow: 0 0 12px rgba(212,160,83,0.08);
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,160,83,0.12) 30%, rgba(212,160,83,0.12) 70%, transparent);
    margin: 16px 0;
}

/* ── Sidebar section labels ── */
.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    color: #3d4455;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 16px 0 8px 0;
    display: flex;
    align-items: center;
    gap: 7px;
    padding-left: 2px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #d4a053 0%, #c7893e 100%);
    color: #0d1017;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.83rem;
    padding: 9px 22px;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(212,160,83,0.2);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e0b06a 0%, #d4943f 100%);
    box-shadow: 0 4px 20px rgba(212,160,83,0.35);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid #1e2436;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    color: #525868;
    font-weight: 500;
    font-size: 0.84rem;
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #9198ab; }
.stTabs [aria-selected="true"] {
    background: #161b26 !important;
    color: #d4a053 !important;
    border-bottom: 2px solid #d4a053;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161b26, #131720);
    border: 1px solid #1e2436;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover { border-color: rgba(212,160,83,0.18); }
[data-testid="stMetricLabel"] p { color: #525868 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"]   { font-family: 'JetBrains Mono', monospace; color: #edeae6; }
[data-testid="stMetricDelta"]   { font-size: 0.75rem !important; }

/* ── Form ── */
.stForm { border: none !important; padding: 0 !important; }



/* ── Containers with border ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #1e2436 !important;
    border-radius: 12px !important;
    background: #13161f !important;
}

/* ── Info / Success alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}
</style>
""", unsafe_allow_html=True)

ICON = {
    "alert":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "check":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "book":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
    "users":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>',
    "trending": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "award":    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
}

if "current_dropout_prob" not in st.session_state:
    st.session_state.current_dropout_prob = None

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

    if st.button("Clear Cache", width='stretch'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Cache cleared.", icon="✓")

    if st.button("Reset Analysis", width='stretch'):
        st.session_state.current_dropout_prob = None
        if "artifact_data" in st.session_state:
            del st.session_state.artifact_data
        st.rerun()

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
        go_btn = st.form_submit_button("Run Analysis", width='stretch')

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
            st.plotly_chart(fig, width='stretch', key=f"radar_{widget_key}")

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
            st.plotly_chart(fg, width='stretch', key=f"gauge_{widget_key}")


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
                r2 = api_post("/predict/dropout", p2)
                if r2 is None:
                    st.error("⚠️ API unreachable. Please check the backend server.")
                elif r2.status_code == 200:
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
            st.plotly_chart(ft, width='stretch', key="artifact_risk_trend")

    with c2:
        if "feature_importance" in data:
            df_f = pd.DataFrame(list(data["feature_importance"].items()), columns=["Feature", "Impact"])
            ff = px.bar(df_f, x="Impact", y="Feature", orientation='h',
                        title="Top Risk Drivers", color_discrete_sequence=['#d4a053'])
            ff.update_layout(yaxis={'categoryorder': 'total ascending'},
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#6b7280'), height=250,
                             margin=dict(t=30, b=10, l=150, r=10))
            st.plotly_chart(ff, width='stretch', key="artifact_feature_importance")

    render_whatif_simulator(data, att, study, payload)


def render_risk_assessment():
    """Render the risk assessment panel directly (no chat interface)."""
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
        r = api_post("/predict/dropout", payload)
        loading_slot.empty()
        if r is None:
            st.error("⚠️ Could not connect to the API server. Make sure it is running (`uvicorn api.main:app --reload`) and try again.")
        elif r.status_code == 200:
            import uuid
            d = r.json()
            d["id"] = str(uuid.uuid4())
            st.session_state.current_dropout_prob = d["dropout_probability"]
            if "artifact_data" not in st.session_state:
                st.session_state.artifact_data = {}
            st.session_state.artifact_data.update(d)
            st.rerun()
        else:
            st.error(f"API returned error {r.status_code}: {r.text[:200]}")

    if "artifact_data" in st.session_state and st.session_state.artifact_data:
        data = st.session_state.artifact_data
        st.markdown("**Analysis complete.** Here is the interactive risk report:")
        render_risk_widgets(data, widget_key=data.get("id", "main"))

        if st.button("Generate Detailed Report", type="primary", key="btn_detailed_report"):
            with st.spinner("Drafting detailed plan..."):
                r_llm = api_post("/predict/analyze", payload)
                if r_llm is not None and r_llm.status_code == 200:
                    llm_data = r_llm.json()
                    st.session_state.artifact_data["reasoning"]    = llm_data["reasoning"]
                    st.session_state.artifact_data["intervention"] = llm_data["intervention"]
                    st.rerun()
                elif r_llm is None:
                    st.warning("⚠️ Could not connect to the API to generate reasoning. Try again.")

        if "reasoning" in st.session_state.artifact_data:
            with st.container(border=True):
                render_artifact_panel(
                    st.session_state.artifact_data,
                    gpa, att, mid, study, miss, payload
                )
    else:
        st.markdown("""
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding:80px 20px; text-align:center; border:1px dashed #1e2436; border-radius:12px; margin-top:20px;">
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#3d4455;
     letter-spacing:3px; text-transform:uppercase; margin:0 0 8px 0;">awaiting input</p>
  <p style="font-family:'Inter',sans-serif; font-size:0.9rem; color:#525868; margin:0;">
    Fill in the student profile in the sidebar and click <strong style="color:#d4a053;">Run Analysis</strong>.
  </p>
</div>
""", unsafe_allow_html=True)


def render_model_comparison():
    try:
        mr = api_get("/metrics")
        if mr is None:
            st.warning("⚠️ Cannot connect to API server to load metrics.")
            return
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
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

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
                st.plotly_chart(fb, width='stretch', key="model_comparison_bar")
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
            st.plotly_chart(fg, width='stretch', key="explorer_grade_dist")

        with c2:
            st.markdown("<p class='t-overline' style='margin-top:16px'>dropout rate by department</p>", unsafe_allow_html=True)
            dr = df.groupby('department')['dropout_risk'].mean().sort_values(ascending=True)
            fd = px.bar(x=dr.values, y=dr.index, labels={'x': '', 'y': ''},
                        orientation='h', color_discrete_sequence=['#c75c5c'])
            fd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#6b7280', family='Inter'), showlegend=False, height=300,
                             xaxis=dict(gridcolor='#1e2230', tickformat='.0%'), yaxis=dict(gridcolor='#1e2230'))
            st.plotly_chart(fd, width='stretch', key="explorer_dropout_dept")

        st.markdown("<p class='t-overline' style='margin-top:16px'>feature correlation matrix</p>", unsafe_allow_html=True)
        ndf  = df.select_dtypes(include=[np.number])
        corr = ndf.corr()
        fcc = px.imshow(corr, text_auto='.2f',
                        color_continuous_scale=[[0, '#1a1e28'], [0.5, '#2a2e3a'], [1, '#d4a053']],
                        zmin=-1, zmax=1, aspect='auto')
        fcc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280', family='Inter', size=8),
                          height=500, margin=dict(t=20, b=20))
        st.plotly_chart(fcc, width='stretch', key="explorer_corr_matrix")

        st.markdown("<p class='t-overline' style='margin-top:16px'>sample records</p>", unsafe_allow_html=True)
        st.dataframe(df.head(80), width='stretch', height=350)

    except FileNotFoundError:
        st.info("No dataset found at data/raw/synthetic_student_data.csv.")


# ── MAIN RENDER ──
if dev_mode:
    t1, t2, t3 = st.tabs(["Risk Assessment", "Model Comparison", "Data Explorer"])
    with t1: render_risk_assessment()
    with t2: render_model_comparison()
    with t3: render_data_explorer()
else:
    render_risk_assessment()

