"""
dashboard/app.py — Credit Risk & Loan Default Dashboard
UI: Brutalist Industrial — concrete grey, amber, left-rail nav via st.columns
"""

import sys, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR     = Path(__file__).parent.parent / "models"
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"

st.set_page_config(page_title="Credit Risk System", page_icon="🏦",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300;400;700;900&family=Roboto+Mono:wght@400;500;700&display=swap');
:root{
    --concrete:#e8e4de; --white:#faf9f7; --black:#1a1814;
    --charcoal:#2e2b26; --amber:#d4820a; --amber2:#f0a020;
    --red:#b91c1c; --green:#16a34a; --muted:#6b6560;
    --border:#bfb9b0; --rail:#1a1814;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
    background:var(--concrete)!important;
    font-family:'Roboto Mono',monospace!important;
    color:var(--black)!important;
}
[data-testid="stSidebar"],[data-testid="collapsedControl"],
button[kind="header"]{display:none!important;}
.main .block-container{max-width:100%!important;padding:8px 20px 20px 20px!important;}

/* System bar */
.sysbar{background:var(--charcoal);color:var(--amber2);
    font-family:'Roboto Mono',monospace;font-size:10px;letter-spacing:.2em;
    padding:7px 20px;border-bottom:2px solid var(--amber);
    display:flex;justify-content:space-between;text-transform:uppercase;
    margin:0 -20px 16px -20px;}

/* Left rail column */
.rail-wrap{background:var(--rail);min-height:90vh;padding:0;
    border-right:3px solid var(--amber);}
.rail-brand{background:var(--amber);padding:16px 14px;
    font-family:'Roboto Slab',serif;font-size:.8rem;font-weight:900;
    color:var(--black);letter-spacing:.05em;text-transform:uppercase;
    line-height:1.3;border-bottom:3px solid var(--black);}
.rail-sect{font-size:8px;letter-spacing:.3em;color:#6b6560;
    text-transform:uppercase;padding:14px 14px 4px 14px;
    font-family:'Roboto Mono',monospace;}
.nav-btn{display:block;width:100%;padding:10px 14px;
    font-family:'Roboto Mono',monospace;font-size:10px;
    color:#c8c4be;background:transparent;border:none;
    border-left:3px solid transparent;text-align:left;
    text-transform:uppercase;letter-spacing:.08em;cursor:pointer;}
.nav-btn:hover,.nav-btn.active{color:var(--amber2)!important;
    border-left-color:var(--amber)!important;
    background:rgba(212,130,10,.1)!important;}
.rail-div{border-top:1px solid #2e2b26;margin:6px 0;}

/* Page content */
h1{font-family:'Roboto Slab',serif!important;font-size:1.7rem!important;
    font-weight:900!important;color:var(--black)!important;
    border-left:5px solid var(--amber)!important;padding-left:14px!important;
    margin-bottom:18px!important;text-transform:uppercase!important;letter-spacing:.03em!important;}
h2{font-family:'Roboto Slab',serif!important;font-size:.95rem!important;
    font-weight:700!important;color:var(--charcoal)!important;
    text-transform:uppercase!important;letter-spacing:.08em!important;}
h3{font-family:'Roboto Mono',monospace!important;font-size:.68rem!important;
    letter-spacing:.25em!important;text-transform:uppercase!important;
    color:var(--muted)!important;}

.kcard{background:var(--white);border:2px solid var(--black);
    border-top:4px solid var(--amber);padding:13px 15px;margin:3px 0;}
.kcard.red{border-top-color:var(--red);}
.kcard.green{border-top-color:var(--green);}
.kcard.black{border-top-color:var(--black);}
.kc-lbl{font-size:9px;letter-spacing:.25em;text-transform:uppercase;
    color:var(--muted);margin-bottom:4px;}
.kc-val{font-family:'Roboto Slab',serif;font-size:1.5rem;
    color:var(--black);font-weight:700;line-height:1.1;}
.kc-sub{font-size:10px;color:var(--muted);margin-top:2px;}

.badge-LOW{background:#dcfce7;color:#166534;border:1px solid #166534;
    padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.1em;}
.badge-MEDIUM{background:#fef9c3;color:#854d0e;border:1px solid #d97706;
    padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.1em;}
.badge-HIGH{background:#fee2e2;color:#991b1b;border:1px solid #991b1b;
    padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.1em;}
.badge-VERY.HIGH{background:#1a1814;color:#f0a020;border:1px solid #d4820a;
    padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.1em;}

div[data-testid="metric-container"]{background:var(--white)!important;
    border:2px solid var(--black)!important;border-top:4px solid var(--amber)!important;
    padding:13px!important;border-radius:0!important;}
div[data-testid="metric-container"] label{font-size:9px!important;
    letter-spacing:.25em!important;text-transform:uppercase!important;
    color:var(--muted)!important;font-family:'Roboto Mono',monospace!important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-family:'Roboto Slab',serif!important;
    font-size:1.5rem!important;color:var(--black)!important;font-weight:700!important;}

.stTabs [data-baseweb="tab"]{font-family:'Roboto Mono',monospace!important;
    font-size:10px!important;letter-spacing:.15em!important;
    text-transform:uppercase!important;color:var(--muted)!important;
    background:transparent!important;border-bottom:3px solid transparent!important;
    padding:9px 16px!important;}
.stTabs [aria-selected="true"]{color:var(--black)!important;
    border-bottom:3px solid var(--amber)!important;}
.stTabs [data-baseweb="tab-list"]{border-bottom:2px solid var(--border)!important;
    background:transparent!important;}
.stButton>button{background:var(--black)!important;color:var(--amber2)!important;
    border:2px solid var(--amber)!important;font-family:'Roboto Mono',monospace!important;
    letter-spacing:.1em!important;text-transform:uppercase!important;
    font-size:11px!important;padding:10px 24px!important;border-radius:0!important;}
.stButton>button:hover{background:var(--amber)!important;color:var(--black)!important;}
[data-testid="stNumberInput"] input,[data-testid="stSelectbox"]>div>div{
    background:var(--white)!important;border:2px solid var(--border)!important;
    border-radius:0!important;font-family:'Roboto Mono',monospace!important;color:var(--black)!important;}
p,li{color:var(--black)!important;}
hr{border:none;border-top:2px solid var(--border);margin:16px 0;}
</style>""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
WHITE="#faf9f7"; BLACK="#1a1814"; AMBER="#d4820a"; AMBER2="#f0a020"
RED="#b91c1c"; GREEN="#16a34a"; MUTED="#6b6560"; CHARCOAL="#2e2b26"
CONCRETE="#e8e4de"

PLOT = dict(
    paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(family="Roboto Mono", color=BLACK, size=11),
    xaxis=dict(gridcolor="#d4cfc8", linecolor="#bfb9b0", tickfont=dict(size=10,color=MUTED)),
    yaxis=dict(gridcolor="#d4cfc8", linecolor="#bfb9b0", tickfont=dict(size=10,color=MUTED)),
    legend=dict(bgcolor=WHITE, bordercolor="#bfb9b0", borderwidth=1, font=dict(size=10)),
    margin=dict(l=50,r=20,t=48,b=40),
)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return (joblib.load(MODEL_DIR/"credit_model.pkl"),
                joblib.load(MODEL_DIR/"scaler.pkl"),
                joblib.load(MODEL_DIR/"feature_names.pkl"))
    except: return None,None,None

@st.cache_data
def load_loans():
    p = RAW_DIR/"loans.csv"
    return pd.read_csv(p, nrows=50000) if p.exists() else None

@st.cache_data
def load_metrics():
    p = MODEL_DIR/"metrics.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data
def load_shap():
    for name in ["shap_importance.csv","feature_importance.csv"]:
        p = MODEL_DIR/name
        if p.exists(): return pd.read_csv(p)
    return None

@st.cache_data
def load_credit_metrics():
    p = MODEL_DIR/"credit_metrics.csv"
    return pd.read_csv(p) if p.exists() else None

model,scaler,feature_names = load_model()
df_loans   = load_loans()
metrics    = load_metrics()
shap_df    = load_shap()
credit_df  = load_credit_metrics()

# ── System bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sysbar">
  <span>■ CREDIT RISK MANAGEMENT SYSTEM v2.0</span>
  <span>MODEL: XGBoost + SMOTE + SHAP &nbsp;|&nbsp; 500K RECORDS &nbsp;|&nbsp; ● OPERATIONAL</span>
</div>""", unsafe_allow_html=True)

# ── Two-column layout: rail + content ─────────────────────────────────────────
rail_col, content_col = st.columns([1, 5])

with rail_col:
    st.markdown("""
    <div class="rail-wrap">
      <div class="rail-brand">CREDIT<br>RISK SYS</div>
      <div class="rail-sect">Navigation</div>
    </div>""", unsafe_allow_html=True)

    pages = ["Portfolio Overview", "Default Analysis",
             "Credit Scoring", "SHAP Explainability", "Risk Metrics"]
    page  = st.radio("", pages, label_visibility="collapsed")

    st.markdown("""
    <style>
    /* Style the radio inside rail col as nav buttons */
    section[data-testid="column"]:first-child [data-testid="stRadio"]>div{
        gap:0!important; background:#1a1814; padding:0;
    }
    section[data-testid="column"]:first-child [data-testid="stRadio"] label{
        display:block!important; width:100%!important;
        padding:10px 14px!important;
        font-family:'Roboto Mono',monospace!important;
        font-size:10px!important; letter-spacing:.08em!important;
        text-transform:uppercase!important;
        color:#c8c4be!important; background:transparent!important;
        border-left:3px solid transparent!important;
        border-radius:0!important; cursor:pointer!important;
    }
    section[data-testid="column"]:first-child [data-testid="stRadio"] label:hover{
        color:#f0a020!important; background:rgba(212,130,10,.1)!important;
    }
    section[data-testid="column"]:first-child [data-testid="stRadio"] [aria-checked="true"] ~ *,
    section[data-testid="column"]:first-child [data-testid="stRadio"] label:has(input:checked){
        color:#f0a020!important; border-left-color:#d4820a!important;
        background:rgba(212,130,10,.1)!important;
    }
    section[data-testid="column"]:first-child [data-testid="stRadio"] [data-baseweb="radio"]{
        display:none!important;
    }
    section[data-testid="column"]:first-child [data-testid="stRadio"] > label{
        display:none!important;
    }
    section[data-testid="column"]:first-child{
        background:#1a1814!important;
        border-right:3px solid #d4820a!important;
        min-height:90vh!important; padding:0!important;
    }
    </style>""", unsafe_allow_html=True)

# ── Content ───────────────────────────────────────────────────────────────────
with content_col:

    # ── PORTFOLIO OVERVIEW ────────────────────────────────────────────────────
    if page == "Portfolio Overview":
        st.title("Portfolio Overview")

        if df_loans is None:
            st.error("Run `python data/generate.py` first.")
        else:
            dr = df_loans["default"].mean()
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f'<div class="kcard black"><div class="kc-lbl">Total Loans</div>'
                        f'<div class="kc-val">{len(df_loans):,}</div></div>',unsafe_allow_html=True)
            c2.markdown(f'<div class="kcard red"><div class="kc-lbl">Default Rate</div>'
                        f'<div class="kc-val">{dr:.2%}</div></div>',unsafe_allow_html=True)
            c3.markdown(f'<div class="kcard"><div class="kc-lbl">Avg Loan Amount</div>'
                        f'<div class="kc-val">${df_loans["loan_amount"].mean():,.0f}</div></div>',
                        unsafe_allow_html=True)
            c4.markdown(f'<div class="kcard green"><div class="kc-lbl">Avg Annual Income</div>'
                        f'<div class="kc-val">${df_loans["annual_inc"].mean():,.0f}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            col1,col2 = st.columns(2)

            with col1:
                st.subheader("Default Rate by Grade")
                gd = df_loans.groupby("grade")["default"].mean().reset_index()
                gd.columns = ["grade","dr"]
                gd["dr"] *= 100
                fig = go.Figure(go.Bar(
                    x=gd["grade"], y=gd["dr"],
                    marker_color=[RED if r>25 else AMBER for r in gd["dr"]],
                    text=gd["dr"].round(1).astype(str)+"%",
                    textposition="outside", textfont=dict(size=10)
                ))
                fig.update_layout(**PLOT, height=300,
                                  title=dict(text="DEFAULT RATE BY LOAN GRADE",
                                             font=dict(size=10,color=MUTED)),
                                  xaxis_title="Grade", yaxis_title="Default Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Loan Volume by Purpose")
                pv = df_loans.groupby("purpose").agg(
                    count=("loan_id","count"),
                    dr=("default","mean")
                ).reset_index().sort_values("count",ascending=True)
                fig2 = go.Figure(go.Bar(
                    x=pv["count"], y=pv["purpose"], orientation="h",
                    marker_color=[RED if r>0.2 else AMBER if r>0.12 else GREEN
                                  for r in pv["dr"]],
                ))
                fig2.update_layout(**PLOT, height=300,
                                   title=dict(text="VOLUME BY PURPOSE (color=default risk)",
                                              font=dict(size=10,color=MUTED)),
                                   xaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Loan Amount by Grade")
            fig3 = go.Figure()
            for g in ["A","B","C","D","E","F","G"]:
                sub = df_loans[df_loans["grade"]==g]["loan_amount"]
                fig3.add_trace(go.Box(y=sub, name=g,
                                       marker_color=AMBER if g in ["A","B","C"] else RED,
                                       boxmean=True))
            fig3.update_layout(**PLOT, height=300,
                               title=dict(text="LOAN AMOUNT DISTRIBUTION BY GRADE",
                                          font=dict(size=10,color=MUTED)),
                               yaxis_title="Loan Amount ($)")
            st.plotly_chart(fig3, use_container_width=True)

    # ── DEFAULT ANALYSIS ──────────────────────────────────────────────────────
    elif page == "Default Analysis":
        st.title("Default Analysis")

        if df_loans is None:
            st.error("No data.")
        else:
            col1,col2 = st.columns(2)
            with col1:
                st.subheader("Default Rate vs DTI")
                bins = pd.cut(df_loans["dti"], bins=10)
                dd = df_loans.groupby(bins, observed=True)["default"].mean().reset_index()
                fig = go.Figure(go.Scatter(
                    x=list(range(len(dd))), y=dd["default"]*100,
                    mode="lines+markers",
                    line=dict(color=AMBER,width=2.5),
                    marker=dict(size=8,color=[RED if v>15 else AMBER
                                for v in dd["default"]*100]),
                    fill="tozeroy", fillcolor="rgba(212,130,10,0.08)"
                ))
                fig.update_layout(**PLOT, height=280,
                                  title=dict(text="DEFAULT RATE vs DTI RATIO",
                                             font=dict(size=10,color=MUTED)),
                                  xaxis_title="DTI Decile", yaxis_title="Default Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Default Rate vs Interest Rate")
                ir_bins = pd.cut(df_loans["int_rate"], bins=10)
                ir = df_loans.groupby(ir_bins, observed=True)["default"].mean().reset_index()
                fig2 = go.Figure(go.Bar(
                    x=list(range(len(ir))), y=ir["default"]*100,
                    marker_color=[RED if v>20 else AMBER for v in ir["default"]*100]
                ))
                fig2.update_layout(**PLOT, height=280,
                                   title=dict(text="DEFAULT RATE vs INTEREST RATE",
                                              font=dict(size=10,color=MUTED)),
                                   xaxis_title="Rate Decile", yaxis_title="Default Rate (%)")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Delinquency vs Default")
            dq = df_loans.groupby("delinq_2yrs")["default"].mean().reset_index().head(8)
            fig3 = go.Figure(go.Bar(
                x=dq["delinq_2yrs"].astype(str), y=dq["default"]*100,
                marker_color=[RED if v>25 else AMBER for v in dq["default"]*100],
                text=(dq["default"]*100).round(1).astype(str)+"%",
                textposition="outside"
            ))
            fig3.update_layout(**PLOT, height=280,
                               title=dict(text="DEFAULT RATE BY DELINQUENCIES (2yr)",
                                          font=dict(size=10,color=MUTED)),
                               xaxis_title="# Delinquencies", yaxis_title="Default Rate (%)")
            st.plotly_chart(fig3, use_container_width=True)

    # ── CREDIT SCORING ────────────────────────────────────────────────────────
    elif page == "Credit Scoring":
        st.title("Live Credit Scorer")

        if model is None:
            st.error("Run `python models/train.py` first.")
        else:
            if metrics:
                m1,m2,m3 = st.columns(3)
                m1.metric("AUC-ROC",       f"{metrics.get('auc_roc','N/A'):.4f}")
                m2.metric("F1 SCORE",      f"{metrics.get('f1','N/A'):.4f}")
                m3.metric("AVG PRECISION", f"{metrics.get('avg_precision','N/A'):.4f}")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Loan Application")

            col1,col2,col3 = st.columns(3)
            with col1:
                loan_amt  = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 500)
                term      = st.selectbox("Term (months)", [36,60])
                int_rate  = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.5, 0.5)
                grade     = st.selectbox("Grade", list("ABCDEFG"), index=2)
            with col2:
                annual_inc = st.number_input("Annual Income ($)", 15000, 300000, 65000, 1000)
                dti        = st.number_input("DTI Ratio (%)", 0.0, 50.0, 18.5, 0.5)
                emp_len    = st.number_input("Employment (years)", 0, 10, 5)
                home_own   = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
            with col3:
                delinq     = st.number_input("Delinquencies (2yr)", 0, 10, 0)
                open_acc   = st.number_input("Open Accounts", 1, 40, 10)
                revol_util = st.number_input("Revolving Util (%)", 0.0, 100.0, 45.0, 1.0)
                credit_hist= st.number_input("Credit History (yrs)", 1, 40, 8)

            if st.button("▶  SCORE APPLICATION"):
                from pipeline.features import engineer
                row = {
                    "loan_amount":loan_amt,"term":term,"int_rate":int_rate,
                    "grade":grade,"emp_length":emp_len,"annual_inc":annual_inc,
                    "dti":dti,"delinq_2yrs":delinq,"open_acc":open_acc,
                    "pub_rec":0,"revol_util":revol_util,"credit_hist_yrs":credit_hist,
                    "purpose":"debt_consolidation","home_ownership":home_own,
                    "loan_to_inc":loan_amt/(annual_inc+1e-6),
                    "installment":loan_amt*int_rate/100/12/(1-(1+int_rate/100/12)**-term),
                }
                df_r = engineer(pd.DataFrame([row]))
                avail= [f for f in feature_names if f in df_r.columns]
                X    = df_r[avail].reindex(columns=feature_names,fill_value=0).fillna(0)
                X_sc = pd.DataFrame(scaler.transform(X), columns=feature_names)
                pd_  = float(model.predict_proba(X_sc)[0][1])
                lgd  = 0.45 + 0.05*list("ABCDEFG").index(grade)
                el   = pd_*lgd*loan_amt
                risk = "LOW" if pd_<0.15 else "MEDIUM" if pd_<0.35 else "HIGH" if pd_<0.60 else "VERY HIGH"
                badge_map = {"LOW":"#16a34a","MEDIUM":"#d97706","HIGH":"#b91c1c","VERY HIGH":"#d4820a"}
                bc = badge_map[risk]

                st.markdown("<hr>", unsafe_allow_html=True)
                r1,r2,r3,r4,r5 = st.columns(5)
                r1.markdown(f'<div class="kcard red"><div class="kc-lbl">PD (Prob Default)</div>'
                            f'<div class="kc-val">{pd_:.2%}</div></div>', unsafe_allow_html=True)
                r2.markdown(f'<div class="kcard"><div class="kc-lbl">LGD</div>'
                            f'<div class="kc-val">{lgd:.2%}</div></div>', unsafe_allow_html=True)
                r3.markdown(f'<div class="kcard"><div class="kc-lbl">EAD</div>'
                            f'<div class="kc-val">${loan_amt:,.0f}</div></div>', unsafe_allow_html=True)
                r4.markdown(f'<div class="kcard red"><div class="kc-lbl">Expected Loss</div>'
                            f'<div class="kc-val">${el:,.0f}</div></div>', unsafe_allow_html=True)
                r5.markdown(f'<div class="kcard"><div class="kc-lbl">Risk Rating</div>'
                            f'<div class="kc-val"><span style="background:{bc};color:white;'
                            f'padding:2px 10px;font-size:11px;font-weight:700">{risk}</span>'
                            f'</div></div>', unsafe_allow_html=True)

                col_g1, col_g2 = st.columns([2,1])
                with col_g1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=pd_*100,
                        title={"text":"Probability of Default (%)","font":{"size":11,"color":MUTED}},
                        number={"suffix":"%","font":{"size":26,"family":"Roboto Slab"}},
                        gauge={
                            "axis":{"range":[0,100],"tickcolor":BLACK},
                            "bar":{"color":RED if pd_>0.35 else AMBER},
                            "steps":[{"range":[0,15],"color":"#dcfce7"},
                                      {"range":[15,35],"color":"#fef9c3"},
                                      {"range":[35,60],"color":"#fee2e2"},
                                      {"range":[60,100],"color":"#fca5a5"}],
                            "threshold":{"line":{"color":RED,"width":4},"value":50}
                        }
                    ))
                    fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                                      height=260, margin=dict(l=30,r=30,t=50,b=10),
                                      font=dict(family="Roboto Mono", color=BLACK))
                    st.plotly_chart(fig, use_container_width=True)

                with col_g2:
                    st.markdown(f"""
                    <div style="background:{WHITE};border:2px solid {BLACK};
                         border-top:4px solid {bc};padding:16px;margin-top:10px;">
                      <div style="font-size:9px;letter-spacing:.25em;text-transform:uppercase;
                           color:{MUTED};margin-bottom:8px">Decision Summary</div>
                      <div style="font-family:'Roboto Slab',serif;font-size:1.1rem;
                           color:{bc};font-weight:700;margin-bottom:12px">{risk} RISK</div>
                      <div style="font-size:11px;color:{BLACK};line-height:1.8">
                        PD &nbsp;&nbsp;&nbsp;: {pd_:.2%}<br>
                        LGD &nbsp;: {lgd:.2%}<br>
                        EAD &nbsp;: ${loan_amt:,.0f}<br>
                        EL &nbsp;&nbsp;&nbsp;: ${el:,.0f}<br>
                        Grade : {grade}
                      </div>
                    </div>""", unsafe_allow_html=True)

    # ── SHAP EXPLAINABILITY ───────────────────────────────────────────────────
    elif page == "SHAP Explainability":
        st.title("SHAP Feature Importance")

        if shap_df is None:
            st.error("Run `python models/train.py` first.")
        else:
            val_col = "mean_shap" if "mean_shap" in shap_df.columns else "importance"
            top20 = shap_df.head(20).sort_values(val_col)

            col1,col2 = st.columns([3,2])
            with col1:
                colors = [RED if i>=15 else AMBER if i>=8 else CHARCOAL
                          for i in range(len(top20))]
                fig = go.Figure(go.Bar(
                    x=top20[val_col], y=top20["feature"],
                    orientation="h", marker_color=colors,
                    text=top20[val_col].round(4), textposition="outside",
                    textfont=dict(size=9,color=BLACK)
                ))
                fig.update_layout(**PLOT, height=520,
                                  title=dict(text="TOP 20 FEATURES — MEAN |SHAP|",
                                             font=dict(size=10,color=MUTED)),
                                  xaxis_title="Mean |SHAP| Value")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Feature Insights")
                insights = {
                    "int_rate":"Higher rate → higher default risk",
                    "dti":"Debt burden vs income",
                    "revol_util":"Credit card usage signal",
                    "loan_to_inc":"Loan size vs capacity",
                    "grade_num":"Lender risk tier",
                    "delinq_2yrs":"Recent missed payments",
                    "annual_inc":"Income stability",
                    "payment_to_inc":"Monthly cashflow burden",
                    "credit_hist_yrs":"Track record length",
                    "open_acc":"Active credit lines",
                }
                for feat,desc in insights.items():
                    st.markdown(f"""
                    <div style="background:{WHITE};border:1px solid {MUTED};
                         border-left:4px solid {AMBER};padding:8px 12px;margin:5px 0;">
                      <div style="font-size:9px;letter-spacing:.2em;color:{MUTED};
                           text-transform:uppercase">{feat}</div>
                      <div style="font-size:11px;color:{BLACK};margin-top:2px">{desc}</div>
                    </div>""", unsafe_allow_html=True)

    # ── RISK METRICS ──────────────────────────────────────────────────────────
    elif page == "Risk Metrics":
        st.title("Credit Risk Metrics")

        if credit_df is not None:
            c1,c2,c3 = st.columns(3)
            c1.metric("AVG PD (Portfolio)",  f"{credit_df['avg_pd'].mean():.2%}")
            c2.metric("TOTAL EXPECTED LOSS", f"${credit_df['expected_loss'].sum():,.0f}")
            c3.metric("HIGHEST RISK GRADE",
                      credit_df.loc[credit_df['avg_pd'].idxmax(),'grade'])

            st.markdown("<hr>", unsafe_allow_html=True)
            col1,col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(name="PD %", x=credit_df["grade"],
                                      y=credit_df["avg_pd"]*100,
                                      marker_color=RED))
                fig.add_trace(go.Scatter(name="LGD %", x=credit_df["grade"],
                                          y=credit_df["lgd"]*100,
                                          mode="lines+markers",
                                          line=dict(color=AMBER,width=2.5),
                                          marker=dict(size=9)))
                fig.update_layout(**PLOT, height=320,
                                  title=dict(text="PD AND LGD BY GRADE",
                                             font=dict(size=10,color=MUTED)),
                                  yaxis_title="Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Bar(
                    x=credit_df["grade"], y=credit_df["expected_loss"],
                    marker_color=[RED if v>500 else AMBER for v in credit_df["expected_loss"]],
                    text=credit_df["expected_loss"].round(0).astype(int),
                    textposition="outside"
                ))
                fig2.update_layout(**PLOT, height=320,
                                   title=dict(text="EXPECTED LOSS PER LOAN BY GRADE",
                                              font=dict(size=10,color=MUTED)),
                                   xaxis_title="Grade", yaxis_title="Expected Loss ($)")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Full Credit Metrics Table")
            disp = credit_df.copy()
            disp["avg_pd"]       = (disp["avg_pd"]*100).round(2).astype(str)+"%"
            disp["lgd"]          = (disp["lgd"]*100).round(2).astype(str)+"%"
            disp["avg_ead"]      = "$"+disp["avg_ead"].round(0).astype(int).astype(str)
            disp["expected_loss"]= "$"+disp["expected_loss"].round(0).astype(int).astype(str)
            st.dataframe(disp, use_container_width=True, height=260)

        if metrics:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Model Performance")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("AUC-ROC",       f"{metrics.get('auc_roc','N/A'):.4f}")
            m2.metric("F1 SCORE",      f"{metrics.get('f1','N/A'):.4f}")
            m3.metric("AVG PRECISION", f"{metrics.get('avg_precision','N/A'):.4f}")
            m4.metric("CV AUC MEAN",   f"{metrics.get('cv_auc_mean','N/A'):.4f}")
            m5.metric("CV AUC STD",    f"±{metrics.get('cv_auc_std','N/A'):.4f}")