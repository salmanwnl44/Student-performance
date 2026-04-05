import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="EduPulse — Student Analytics", layout="wide", page_icon="◉", initial_sidebar_state="expanded")

# ── DESIGN SYSTEM ──────────────────────────────────────────
# Palette: Deep slate + warm amber/gold + muted sage green + soft coral
# Typography: Inter for body, monospace accents for data
# Philosophy: editorial, data-dense, Bloomberg-terminal-meets-SaaS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Base ── */
    .stApp {
        background: #12151c;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: #181c25;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    h1, h2, h3, h4 { color: #e8e6e3 !important; font-weight: 600; }

    /* ── Cards ── */
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
    .card-accent-sage { border-left: 3px solid #6b9e7a; }
    .card-accent-blue { border-left: 3px solid #5b7fb5; }

    /* ── Typography ── */
    .t-overline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .t-display {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }
    .t-body { color: #9ca3af; font-size: 0.88rem; line-height: 1.55; }
    .t-caption { color: #6b7280; font-size: 0.78rem; }
    .t-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #d4a053;
    }

    /* ── Colors ── */
    .c-amber { color: #d4a053; }
    .c-coral { color: #c75c5c; }
    .c-sage { color: #6b9e7a; }
    .c-blue { color: #5b7fb5; }
    .c-ghost { color: #4b5563; }
    .c-light { color: #e8e6e3; }

    /* ── Status Pills ── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .pill-danger { background: #2a1a1a; color: #c75c5c; border: 1px solid #3d2424; }
    .pill-safe { background: #1a2a1e; color: #6b9e7a; border: 1px solid #243d28; }
    .pill-warn { background: #2a2618; color: #d4a053; border: 1px solid #3d3522; }

    /* ── Dividers ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #24293a 20%, #24293a 80%, transparent);
        margin: 20px 0;
    }

    /* ── Button ── */
    .stButton>button {
        background: #d4a053;
        color: #12151c;
        border: none;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 10px 28px;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #e0b06a;
        box-shadow: 0 4px 16px rgba(212,160,83,0.25);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #24293a; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px 6px 0 0;
        padding: 10px 22px;
        color: #6b7280;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1e28 !important;
        color: #d4a053 !important;
        border-bottom: 2px solid #d4a053;
    }

    /* ── Metrics override ── */
    [data-testid="stMetric"] {
        background: #1a1e28;
        border: 1px solid #24293a;
        border-radius: 8px;
        padding: 14px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #e8e6e3;
    }

    /* Fix form spacing */
    .stForm { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── SVG ICONS ──
ICON = {
    "shield": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "alert": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "check": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "book": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
    "clock": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "target": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "users": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>',
    "trending": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "wifi": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12.55a11 11 0 0114.08 0"/><path d="M1.42 9a16 16 0 0121.16 0"/><path d="M8.53 16.11a6 6 0 016.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>',
    "award": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "briefcase": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/></svg>',
}

# ── HEADER ──
st.markdown("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
    <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #d4a053, #c7893e); border-radius: 8px; display: flex; align-items: center; justify-content: center;">
        <span style="font-size: 1.2rem; font-weight: 900; color: #12151c;">E</span>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 1.6rem; letter-spacing: -0.5px;">EduPulse</h1>
        <p class="t-caption" style="margin: 0;">Student analytics & intervention engine</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Comparison", "Data Explorer"])

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("<p class='t-overline'>student profile input</p>", unsafe_allow_html=True)
    with st.form("f"):
        st.markdown(f"<p class='t-body' style='margin-bottom:8px'>{ICON['users']} <strong class='c-light'>Demographics</strong></p>", unsafe_allow_html=True)
        age = st.number_input("Age", 15, 100, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
        dept = st.selectbox("Department", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Medicine"])
        sem = st.slider("Semester", 1, 8, 3)
        pedu = st.selectbox("Parent Education", ["No Formal Education", "High School", "Bachelor", "Master", "PhD"])
        finc = st.selectbox("Family Income", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
        commute = st.slider("Commute (mi)", 0.0, 80.0, 10.0)
        schol = st.checkbox("Scholarship")
        inet = st.checkbox("Internet Access", value=True)
        job = st.checkbox("Part-Time Job")

        st.markdown(f"<div class='divider'></div><p class='t-body'>{ICON['book']} <strong class='c-light'>Academics</strong></p>", unsafe_allow_html=True)
        gpa = st.slider("GPA", 0.0, 4.0, 2.8, 0.1)
        att = st.slider("Attendance %", 0.0, 100.0, 78.0)
        mid = st.slider("Midterm", 0.0, 100.0, 65.0)
        asgn = st.slider("Assignments", 0.0, 100.0, 70.0)
        quiz = st.slider("Quizzes", 0.0, 100.0, 60.0)
        miss = st.number_input("Missed Deadlines", 0, 50, 2)

        st.markdown(f"<div class='divider'></div><p class='t-body'>{ICON['trending']} <strong class='c-light'>Engagement</strong></p>", unsafe_allow_html=True)
        logins = st.number_input("Weekly Logins", 0, 100, 8)
        forum = st.number_input("Forum Posts", 0, 50, 3)
        tmat = st.slider("LMS Hours/Wk", 0.0, 60.0, 10.0)
        study = st.slider("Study Hours/Wk", 0.0, 40.0, 8.0)
        lib = st.number_input("Library Visits/Wk", 0, 30, 2)

        go_btn = st.form_submit_button("Run Analysis")

payload = {"age":age,"gender":gender,"department":dept,"semester":sem,"parent_education":pedu,"family_income":finc,
    "commute_distance":commute,"scholarship_status":int(schol),"has_internet_access":int(inet),"part_time_job":int(job),
    "gpa_current":gpa,"attendance_rate":att,"midterm_score":mid,"assignment_score":asgn,"quiz_score":quiz,
    "login_frequency":logins,"forum_participation_score":forum,"time_spent_on_materials":tmat,
    "study_hours_weekly":study,"library_usage_weekly":lib,"missed_deadlines_count":miss}

# ════════════════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════
with tab1:
    if go_btn:
        try:
            r = requests.post(f"{API_BASE}/predict/dropout", json=payload)
            if r.status_code == 200:
                d = r.json()
                risk = d["student_risk_assessment"]
                prob = d["dropout_probability"]

                # ── Metric Strip ──
                m1,m2,m3,m4,m5 = st.columns(5)
                with m1:
                    pill_cls = "pill-danger" if risk=="High" else "pill-safe"
                    icon = ICON["alert"] if risk=="High" else ICON["check"]
                    st.markdown(f"<div class='card'><p class='t-overline'>status</p><span class='pill {pill_cls}'>{icon} {risk} Risk</span></div>", unsafe_allow_html=True)
                with m2:
                    pc = "c-coral" if prob>0.5 else "c-amber" if prob>0.2 else "c-sage"
                    st.markdown(f"<div class='card'><p class='t-overline'>dropout probability</p><p class='t-display {pc}'>{prob:.1%}</p></div>", unsafe_allow_html=True)
                with m3:
                    gc = "c-coral" if gpa<2.0 else "c-amber" if gpa<3.0 else "c-sage"
                    st.markdown(f"<div class='card'><p class='t-overline'>gpa</p><p class='t-display {gc}'>{gpa:.2f}</p></div>", unsafe_allow_html=True)
                with m4:
                    ac = "c-coral" if att<70 else "c-amber" if att<85 else "c-sage"
                    st.markdown(f"<div class='card'><p class='t-overline'>attendance</p><p class='t-display {ac}'>{att:.0f}%</p></div>", unsafe_allow_html=True)
                with m5:
                    mc2 = "c-coral" if mid<50 else "c-amber" if mid<70 else "c-sage"
                    st.markdown(f"<div class='card'><p class='t-overline'>midterm</p><p class='t-display {mc2}'>{mid:.0f}</p></div>", unsafe_allow_html=True)

                # ── Charts ──
                cr, cg = st.columns([3, 2])
                with cr:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='t-overline'>performance profile</p>", unsafe_allow_html=True)
                    cats = ['Attendance','GPA','Midterm','Logins','Study','Deadlines']
                    sv = [att/10,(gpa/4)*10,mid/10,min(logins,10),min(study/2,10),max(10-miss*1.5,0)]
                    bv = [8.5,7.5,7,7,7,9]
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=bv,theta=cats,fill='toself',name='Cohort Avg',
                        line=dict(color='#3a3f52',width=1),fillcolor='rgba(58,63,82,0.15)'))
                    fig.add_trace(go.Scatterpolar(r=sv,theta=cats,fill='toself',name='This Student',
                        line=dict(color='#d4a053',width=2),fillcolor='rgba(212,160,83,0.12)'))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,10],color="#2a2e3a",gridcolor="#24293a"),
                        angularaxis=dict(color="#6b7280",gridcolor="#24293a"),bgcolor='rgba(0,0,0,0)'),
                        showlegend=True,legend=dict(font=dict(color='#6b7280',size=11)),
                        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#6b7280',family='Inter'),height=370,margin=dict(t=50,b=30,l=50,r=50))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with cg:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<p class='t-overline'>risk meter</p>", unsafe_allow_html=True)
                    bar_col = "#c75c5c" if prob>0.5 else "#d4a053" if prob>0.2 else "#6b9e7a"
                    fg = go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
                        number=dict(suffix="%",font=dict(color='#e8e6e3',size=48,family='JetBrains Mono')),
                        gauge=dict(axis=dict(range=[0,100],tickcolor='#2a2e3a',tickfont=dict(color='#4b5563')),
                            bar=dict(color=bar_col,thickness=0.7),
                            bgcolor='#1a1e28',borderwidth=0,
                            steps=[dict(range=[0,20],color='#1a2a1e'),dict(range=[20,50],color='#2a2618'),dict(range=[50,100],color='#2a1a1a')])))
                    fg.update_layout(paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7280'),height=300,margin=dict(t=50,b=20,l=30,r=30))
                    st.plotly_chart(fg, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── Intervention Plan ──
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown(f"<p class='t-overline' style='margin-bottom:12px'>{ICON['target']} intervention plan</p>", unsafe_allow_html=True)

                recs = []
                if att<85: recs.append(("Attendance Recovery",f"Currently at {att:.0f}%. Attend every session for the next 30 days to cross the 85% safety threshold. Consider front-row seating to boost focus.","coral","alert"))
                if gpa<2.5: recs.append(("Academic Remediation",f"GPA of {gpa:.1f} flags academic distress. Arrange 3 hrs/week peer tutoring. Prioritise the lowest-scoring subject first.","coral","book"))
                if mid<60: recs.append(("Midterm Recovery",f"Score of {mid:.0f}/100. Review past midterm papers with a study group. Schedule 2 office-hour visits before the next exam.","amber","target"))
                if study<10: recs.append(("Study Routine",f"Logging {study:.0f} hrs/wk. Adopt structured blocks: 3 × 90 min deep-focus sessions with 20 min breaks.","amber","clock"))
                if miss>1: recs.append(("Deadline Discipline",f"{miss} submissions missed. Set alerts at 72h, 48h, and 12h before each deadline. Break large tasks into daily sub-tasks.","amber","alert"))
                if forum<3: recs.append(("Discussion Engagement","Posting ≥ 2 substantive responses per week correlates with 18% higher course completion. Start with questions about lecture material.","blue","users"))
                if job and gpa<3.0: recs.append(("Work-Study Balance","Part-time employment is measurably impacting academics. Reduce shifts during exam weeks or explore on-campus work-study options.","amber","briefcase"))
                if not inet: recs.append(("Digital Access","No home internet detected. Campus library offers free broadband daily 7am–11pm. Download lecture materials for offline review.","blue","wifi"))
                if not schol and gpa>=3.0: recs.append(("Merit Scholarship","Current GPA qualifies for departmental merit aid. Apply before the semester deadline to offset financial barriers.","sage","award"))

                if not recs:
                    st.markdown(f"<div class='card card-accent-sage'><p class='t-body'>{ICON['check']} <strong class='c-sage'>All clear.</strong> This student's profile is healthy across all indicators. Encourage them to maintain current routines.</p></div>", unsafe_allow_html=True)
                else:
                    for title, text, accent, icon_key in recs:
                        ic = ICON.get(icon_key, "")
                        st.markdown(f"""<div class='card card-accent-{accent}'>
                            <p style='margin:0 0 4px 0;color:#e8e6e3;font-weight:600;font-size:0.9rem'>{ic} {title}</p>
                            <p class='t-body' style='margin:0'>{text}</p>
                        </div>""", unsafe_allow_html=True)
            else:
                st.error(f"Server error: {r.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    else:
        st.markdown(f"""<div class='card' style='text-align:center;padding:80px 40px'>
            <div style='margin-bottom:16px'>{ICON['shield'].replace('18','48').replace('stroke-width="2"','stroke-width="1.5" stroke="#3a3f52"')}</div>
            <h3 style='margin:0 0 8px 0'>Ready for Analysis</h3>
            <p class='t-body' style='max-width:420px;margin:0 auto'>Configure a student profile using the sidebar controls, then click <strong class='c-amber'>Run Analysis</strong> to generate risk scores and actionable recommendations.</p>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════
with tab2:
    try:
        mr = requests.get(f"{API_BASE}/metrics")
        if mr.status_code == 200:
            md = mr.json()
            best = md.get("best_model","rf")
            ds = md.get("dataset_size","?")
            fc = md.get("features_count","?")

            st.markdown(f"<p class='t-overline'>training summary</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><p class='t-body'>{ICON['award']} Champion: <strong class='c-amber'>{md.get(best,{}).get('name',best)}</strong> &nbsp;·&nbsp; <span class='t-mono'>{ds:,}</span> samples &nbsp;·&nbsp; <span class='t-mono'>{fc}</span> features</p></div>", unsafe_allow_html=True)

            rows = []
            for k in ['rf','xgb','lgbm']:
                if k in md:
                    m = md[k]['metrics']
                    rows.append({'Model':md[k]['name'],'Accuracy':f"{m['accuracy']:.2%}",'Precision':f"{m['precision']:.2%}",
                        'Recall':f"{m['recall']:.2%}",'F1':f"{m['f1_score']:.2%}",
                        'AUC':f"{m['roc_auc']:.4f}" if isinstance(m['roc_auc'],float) else m['roc_auc'],
                        ' ':'◆' if k==best else ''})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Bar comparison
            bd = []
            for k in ['rf','xgb','lgbm']:
                if k in md:
                    m=md[k]['metrics']; n=md[k]['name']
                    for mn in ['accuracy','precision','recall','f1_score']:
                        bd.append({'Model':n,'Metric':mn.replace('_',' ').title(),'Value':m[mn]})
            if bd:
                fb=px.bar(pd.DataFrame(bd),x='Metric',y='Value',color='Model',barmode='group',
                    color_discrete_sequence=['#5b7fb5','#d4a053','#6b9e7a'])
                fb.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#6b7280',family='Inter'),
                    yaxis=dict(range=[0.9,1.0],gridcolor='#1e2230',title=''),
                    xaxis=dict(gridcolor='#1e2230',title=''),height=380,
                    legend=dict(font=dict(color='#9ca3af')))
                st.plotly_chart(fb,use_container_width=True)

            # Confusion matrices
            st.markdown("<p class='t-overline' style='margin-top:20px'>confusion matrices</p>", unsafe_allow_html=True)
            cc=st.columns(3)
            for i,k in enumerate(['rf','xgb','lgbm']):
                if k in md:
                    cm=md[k]['metrics'].get('confusion_matrix',[[0,0],[0,0]])
                    with cc[i]:
                        st.markdown(f"<p class='t-caption' style='text-align:center'>{md[k]['name']}</p>", unsafe_allow_html=True)
                        fc2=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual"),
                            x=['Safe','At-Risk'],y=['Safe','At-Risk'],color_continuous_scale=[[0,'#1a1e28'],[1,'#d4a053']])
                        fc2.update_layout(paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7280',family='Inter',size=11),
                            height=260,margin=dict(t=20,b=10),coloraxis_showscale=False)
                        st.plotly_chart(fc2,use_container_width=True)
        else:
            st.info("No training data available yet.")
    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")

# ════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ════════════════════════════════════════════════════════
with tab3:
    try:
        df = pd.read_csv("data/raw/synthetic_student_data.csv")
        st.markdown("<p class='t-overline'>dataset overview</p>", unsafe_allow_html=True)

        s1,s2,s3,s4,s5 = st.columns(5)
        with s1: st.metric("Records", f"{len(df):,}")
        with s2: st.metric("Avg GPA", f"{df['gpa_current'].mean():.2f}")
        with s3: st.metric("Attendance", f"{df['attendance_rate'].mean():.1f}%")
        with s4: st.metric("Dropout Rate", f"{df['dropout_risk'].mean():.1%}")
        with s5: st.metric("Missing Cells", f"{df.isnull().sum().sum():,}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<p class='t-overline' style='margin-top:16px'>grade distribution</p>", unsafe_allow_html=True)
            gc = df['final_grade'].value_counts().reindex(['A','B','C','D','F'])
            fg = px.bar(x=gc.index,y=gc.values,labels={'x':'','y':''},
                color_discrete_sequence=['#d4a053'])
            fg.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#6b7280',family='Inter'),showlegend=False,height=300,
                xaxis=dict(gridcolor='#1e2230'),yaxis=dict(gridcolor='#1e2230'))
            st.plotly_chart(fg,use_container_width=True)

        with c2:
            st.markdown("<p class='t-overline' style='margin-top:16px'>dropout rate by department</p>", unsafe_allow_html=True)
            dr = df.groupby('department')['dropout_risk'].mean().sort_values(ascending=True)
            fd = px.bar(x=dr.values,y=dr.index,labels={'x':'','y':''},orientation='h',
                color_discrete_sequence=['#c75c5c'])
            fd.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#6b7280',family='Inter'),showlegend=False,height=300,
                xaxis=dict(gridcolor='#1e2230',tickformat='.0%'),yaxis=dict(gridcolor='#1e2230'))
            st.plotly_chart(fd,use_container_width=True)

        st.markdown("<p class='t-overline' style='margin-top:16px'>feature correlation matrix</p>", unsafe_allow_html=True)
        ndf = df.select_dtypes(include=[np.number])
        corr = ndf.corr()
        fcc = px.imshow(corr,text_auto='.2f',color_continuous_scale=[[0,'#1a1e28'],[0.5,'#2a2e3a'],[1,'#d4a053']],zmin=-1,zmax=1,aspect='auto')
        fcc.update_layout(paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7280',family='Inter',size=8),height=500,margin=dict(t=20,b=20))
        st.plotly_chart(fcc, use_container_width=True)

        st.markdown("<p class='t-overline' style='margin-top:16px'>sample records</p>", unsafe_allow_html=True)
        st.dataframe(df.head(80), use_container_width=True, height=350)
    except FileNotFoundError:
        st.info("No dataset found.")
