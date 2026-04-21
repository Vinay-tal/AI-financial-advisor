"""
FinSight India — AI Financial Behavior Engine
Indian UPI + Bank Statement Analyzer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, io

sys.path.insert(0, os.path.dirname(__file__))

from utils.parsers.sms_parser import parse_sms_file, parse_sms_dataframe
from utils.parsers.bank_csv_parser import parse_bank_csv, generate_sample_sms
from utils.india_processor import (
    get_india_monthly_summary, get_india_category_breakdown,
    get_india_spending_trends, compute_india_risk_features,
    generate_india_recommendations, benchmark_comparison,
    INDIA_CATEGORY_COLORS, INDIA_CATEGORY_ICONS, INDIA_BENCHMARKS,
)
from utils.risk_model import train_model, predict_risk, get_feature_importance, FEATURE_COLS
from utils.ai_advisor import get_ai_response

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight India — UPI Financial Analyzer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg: #080f1a; --card: #0f1d2e; --card2: #162236;
    --saffron: #ff9933; --white: #f8f8f8; --green: #138808;
    --accent: #ff9933; --blue: #1a73e8; --purple: #7c3aed;
    --text: #e8edf5; --muted: #7a8fa6; --border: rgba(255,255,255,0.07);
}
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif!important; background:var(--bg)!important; color:var(--text)!important; }
.stApp{ background:var(--bg)!important; }
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0a1526,#080f1a)!important; border-right:1px solid var(--border)!important; }

.kpi{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; position:relative; overflow:hidden; }
.kpi::after{ content:''; position:absolute; bottom:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--saffron),var(--green)); }
.kpi-label{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px; }
.kpi-val{ font-size:26px; font-weight:700; font-family:'DM Mono'; color:var(--text); }
.kpi-sub{ font-size:12px; margin-top:5px; color:var(--muted); }
.pos{ color:#22c55e; } .neg{ color:#ef4444; } .warn{ color:#f59e0b; }

.bench-card{ background:var(--card); border-radius:12px; padding:14px 18px; margin-bottom:10px; border-left:3px solid var(--border); }
.bench-high{ border-left-color:#ef4444; }
.bench-above{ border-left-color:#f59e0b; }
.bench-normal{ border-left-color:#22c55e; }
.bench-low{ border-left-color:#3b82f6; }

.rec-card{ background:var(--card); border-left:3px solid var(--blue); border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; }
.rec-card.High{ border-left-color:#ef4444; }
.rec-card.Medium{ border-left-color:#f59e0b; }
.rec-card.Low{ border-left-color:#22c55e; }

.chat-user{ background:linear-gradient(135deg,#1a3a5c,#152d47); border-radius:16px 16px 4px 16px; padding:12px 16px; margin:8px 0; margin-left:12%; font-size:14px; border:1px solid rgba(26,115,232,0.2); }
.chat-ai{ background:var(--card2); border-radius:16px 16px 16px 4px; padding:12px 16px; margin:8px 0; margin-right:12%; font-size:14px; border:1px solid var(--border); }
.chat-label{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:var(--muted); margin-bottom:5px; }

.flag-header{ background:linear-gradient(135deg,#ff9933,#ffffff,#138808); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:32px; font-weight:800; }

#MainMenu,footer,.stDeployButton,div[data-testid="stToolbar"]{ visibility:hidden; display:none; }
.stTabs [data-baseweb="tab-list"]{ background:transparent!important; border-bottom:1px solid var(--border)!important; }
.stTabs [data-baseweb="tab"]{ background:transparent!important; color:var(--muted)!important; font-family:'DM Sans'!important; font-weight:500!important; }
.stTabs [aria-selected="true"]{ background:var(--card2)!important; color:var(--text)!important; border-bottom:2px solid var(--saffron)!important; }
.stButton>button{ background:linear-gradient(135deg,#ff9933,#e8850d)!important; color:#000!important; font-weight:700!important; border:none!important; border-radius:10px!important; font-family:'DM Sans'!important; }
.stTextInput>div>div>input,.stTextArea>div>div>textarea{ background:var(--card2)!important; border:1px solid var(--border)!important; border-radius:10px!important; color:var(--text)!important; }
.stSelectbox>div>div{ background:var(--card2)!important; border:1px solid var(--border)!important; border-radius:10px!important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#7a8fa6"),
    margin=dict(l=10,r=10,t=35,b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
)

# ── Session state ─────────────────────────────────────────────
for k, v in [("df",None),("model",None),("risk_result",{}),("chat_history",[]),
              ("features_df",None),("bank_name",""),("data_source","")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
        <span class="flag-header">💳 FinSight India</span>
        <div style="font-size:11px;color:#7a8fa6;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">UPI · Bank Statement · AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio("📥 Data Source", ["🏦 Bank Statement CSV", "📱 SMS Transactions", "🎯 Demo Data"],
                    index=2, label_visibility="collapsed")
    st.markdown(f"**{mode}**")

    uploaded = None
    sms_text = None

    if mode == "🏦 Bank Statement CSV":
        st.markdown("Upload your bank statement CSV")
        st.markdown("""<div style='font-size:11px;color:#7a8fa6;line-height:1.7;padding:8px;background:rgba(255,255,255,0.03);border-radius:8px;'>
        ✅ HDFC Bank<br>✅ SBI<br>✅ ICICI Bank<br>✅ Axis Bank<br>✅ Kotak<br>✅ YES Bank<br>✅ Paytm/PhonePe exports
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        st.caption("Download from: Net Banking → Statements → Export CSV")

    elif mode == "📱 SMS Transactions":
        st.markdown("Paste SMS messages below")
        st.caption("Export SMS using 'SMS Backup & Restore' app → copy text")
        sms_text = st.text_area("Paste SMS messages (one per line or paragraph)",
                                 height=200, label_visibility="collapsed",
                                 placeholder="Dear Customer, INR 450.00 debited from A/c XX5678 on 21-04-2024 to VPA zomato@icici...")

    else:  # Demo
        demo_choice = st.selectbox("Select demo bank",
                                    ["HDFC Bank Statement", "SBI Bank Statement", "SMS Transactions"])

    st.markdown("---")
    if st.button("🔍 Analyze My Finances", use_container_width=True):
        with st.spinner("Analyzing your financial data..."):
            df = None
            bank = ""
            err = ""
            try:
                if mode == "🏦 Bank Statement CSV" and uploaded:
                    df, bank, err = parse_bank_csv(uploaded)
                    source = f"Bank CSV ({bank})"

                elif mode == "📱 SMS Transactions" and sms_text and sms_text.strip():
                    df = parse_sms_file(sms_text)
                    bank = "SMS"
                    source = "SMS Parser"

                else:  # Demo
                    base = os.path.join(os.path.dirname(__file__), "data", "samples")
                    if demo_choice == "HDFC Bank Statement":
                        df, bank, err = parse_bank_csv(os.path.join(base, "hdfc_statement.csv"))
                    elif demo_choice == "SBI Bank Statement":
                        df, bank, err = parse_bank_csv(os.path.join(base, "sbi_statement.csv"))
                    else:
                        sms_demo = generate_sample_sms()
                        df = parse_sms_file(sms_demo)
                        bank = "SMS Demo"
                    source = f"Demo ({demo_choice})"

                if err:
                    st.error(f"Parse error: {err}")
                elif df is None or df.empty:
                    st.error("No transactions found. Check format.")
                else:
                    st.session_state.df = df
                    st.session_state.bank_name = bank
                    st.session_state.data_source = source

                    features_df = compute_india_risk_features(df)
                    st.session_state.features_df = features_df

                    if not features_df.empty:
                        model = train_model(features_df)
                        st.session_state.model = model
                        risk = predict_risk(model, features_df.iloc[-1].to_dict())
                        st.session_state.risk_result = risk

                    st.success(f"✅ {len(df):,} transactions loaded from {bank}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.markdown("---")
    groq_key = st.text_input("🤖 Groq API Key", type="password",
                              placeholder="gsk_... (free at console.groq.com)",
                              label_visibility="visible")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        import utils.ai_advisor as ai_mod
        ai_mod.GROQ_API_KEY = groq_key

    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#475569;line-height:2;'>
    <b style='color:#64748b;'>SMS EXPORT GUIDE:</b><br>
    Android: SMS Backup & Restore<br>
    → Backup as XML/text<br>
    → Copy bank SMS messages<br>
    → Paste in SMS box above<br><br>
    <b style='color:#64748b;'>BANK CSV GUIDE:</b><br>
    HDFC: NetBanking → Accounts<br>
    → Last 6 months → Download<br>
    SBI: OnlineSBI → Statements<br>
    ICICI: iMobile → Statements
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:60px;margin-bottom:20px;">🇮🇳</div>
        <h1 style="font-size:42px;font-weight:800;background:linear-gradient(135deg,#ff9933,#ffffff,#138808);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;">
            FinSight India
        </h1>
        <p style="font-size:18px;color:#7a8fa6;max-width:560px;margin:0 auto 36px;line-height:1.6;">
            AI-powered financial analysis built for Indian households.<br>
            Works with UPI SMS alerts & all major bank statements.
        </p>
        <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:48px;">
    """, unsafe_allow_html=True)

    for icon, title, desc in [
        ("📱","UPI SMS Parser","Parses HDFC, SBI, ICICI, Axis, PhonePe, GPay SMS alerts"),
        ("🏦","Bank CSV Import","Supports all major Indian bank statement formats"),
        ("🤖","Risk Prediction","ML model trained on your spending patterns"),
        ("🆚","India Benchmarks","Compare vs avg Indian household spending"),
        ("💬","AI Advisor","Chat in context of your real transactions"),
        ("💡","Smart Recs","SIP nudges, EMI alerts, UPI micro-spend tracking"),
    ]:
        st.markdown(f"""
        <div style="background:#0f1d2e;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px 22px;min-width:160px;max-width:180px;display:inline-block;margin:4px;text-align:left;">
            <div style="font-size:26px;margin-bottom:8px;">{icon}</div>
            <div style="font-weight:600;font-size:14px;margin-bottom:4px;">{title}</div>
            <div style="font-size:12px;color:#7a8fa6;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        </div>
        <p style="color:#ff9933;font-weight:600;font-size:15px;">← Choose a data source in the sidebar and click Analyze</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
df = st.session_state.df
model = st.session_state.model
risk_result = st.session_state.risk_result
features_df = st.session_state.features_df
bank_name = st.session_state.bank_name

monthly = get_india_monthly_summary(df)
cat_breakdown = get_india_category_breakdown(df)

# Header
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:24px;">
    <div>
        <div style="font-size:24px;font-weight:800;">🇮🇳 FinSight India Dashboard</div>
        <div style="font-size:13px;color:#7a8fa6;margin-top:2px;">{st.session_state.data_source} · {len(df):,} transactions · {df['date'].min().strftime('%d %b %Y')} – {df['date'].max().strftime('%d %b %Y')}</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:13px;font-weight:600;color:#ff9933;">{bank_name}</div>
        <div style="font-size:11px;color:#7a8fa6;">Detected Bank</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔮 Risk & EMI", "🆚 India Benchmarks", "💡 Recommendations", "💬 AI Advisor"])


# ═══════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════
with tab1:
    latest = monthly.iloc[-1] if not monthly.empty else {}
    income = latest.get("total_income", 0)
    expenses = latest.get("total_expenses", 0)
    net = latest.get("net_flow", 0)
    sr = latest.get("savings_rate", 0)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, sub, cls in [
        (c1, "Monthly Income",   f"₹{income:,.0f}",   "💰 Credits",    "pos"),
        (c2, "Monthly Expenses", f"₹{expenses:,.0f}", "💸 Debits",     "neg"),
        (c3, "Net Savings",      f"₹{net:,.0f}",      "▲ Surplus" if net>=0 else "▼ Deficit", "pos" if net>=0 else "neg"),
        (c4, "Savings Rate",     f"{sr:.1f}%",         "🎯 Target: 20%","pos" if sr>=20 else "warn" if sr>=10 else "neg"),
        (c5, "Risk Level",       f"{risk_result.get('emoji','⚪')} {risk_result.get('risk_level','—')}",
                                                       f"{risk_result.get('risk_probability',0)*100:.0f}% probability", ""),
    ]:
        col.markdown(f"""<div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-sub {cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3,2])

    with col_left:
        st.markdown("#### 📈 Income vs Expenses")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Income", x=monthly["month"], y=monthly["total_income"],
                             marker_color="#22c55e", opacity=0.85))
        fig.add_trace(go.Bar(name="Expenses", x=monthly["month"], y=monthly["total_expenses"],
                             marker_color="#ef4444", opacity=0.85))
        fig.add_trace(go.Scatter(name="Net", x=monthly["month"], y=monthly["net_flow"],
                                 mode="lines+markers", line=dict(color="#ff9933", width=2.5),
                                 marker=dict(size=7)))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=300,
                          legend=dict(orientation="h",y=1.1),
                          yaxis_title="Amount (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🏷️ Spending Breakdown")
        colors = [INDIA_CATEGORY_COLORS.get(c,"#94a3b8") for c in cat_breakdown["category"]]
        fig2 = go.Figure(go.Pie(
            labels=cat_breakdown["category"], values=cat_breakdown["abs_amount"],
            hole=0.55, marker=dict(colors=colors, line=dict(color="#080f1a",width=2)),
            textinfo="percent+label", textfont=dict(size=11),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Category bars
    col3a, col3b = st.columns([3,2])
    with col3a:
        st.markdown("#### 📉 Spending Trends")
        trend = get_india_spending_trends(df)
        top_cats = cat_breakdown.head(6)["category"].tolist()
        trend_f = trend[trend["category"].isin(top_cats)]
        fig3 = px.line(trend_f, x="month", y="abs_amount", color="category",
                       color_discrete_map=INDIA_CATEGORY_COLORS, markers=True)
        fig3.update_traces(line=dict(width=2))
        fig3.update_layout(**PLOTLY_LAYOUT, height=280, yaxis_title="₹", xaxis_title="",
                           legend=dict(orientation="h",y=1.12,font=dict(size=10)))
        st.plotly_chart(fig3, use_container_width=True)

    with col3b:
        st.markdown("#### 📋 Category Summary")
        for _, row in cat_breakdown.iterrows():
            icon = INDIA_CATEGORY_ICONS.get(row["category"],"📦")
            color = INDIA_CATEGORY_COLORS.get(row["category"],"#94a3b8")
            pct = row["percentage"]
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-size:13px;">{icon} {row['category']}</span>
                    <span style="font-size:13px;font-family:'DM Mono';color:{color};">₹{row['abs_amount']:,.0f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;">
                    <div style="width:{int(pct)}%;height:100%;border-radius:4px;background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Transactions table
    st.markdown("#### 🧾 Recent Transactions")
    show_cols = [c for c in ["date","description","category","amount","balance","bank"] if c in df.columns]
    recent = df[show_cols].tail(30).copy()
    recent["date"] = recent["date"].dt.strftime("%d %b %Y")
    recent["amount"] = recent["amount"].apply(lambda x: f"+₹{x:,.2f}" if x>0 else f"-₹{abs(x):,.2f}")
    st.dataframe(recent.iloc[::-1], use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════
# TAB 2: RISK & EMI
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔮 Financial Risk & EMI Analysis")
    rp = risk_result.get("risk_probability", 0)
    rl = risk_result.get("risk_level", "Unknown")

    col_g, col_emi, col_detail = st.columns([1,1,2])

    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rp*100,
            title={"text":"Risk Score","font":{"color":"#7a8fa6","size":13}},
            number={"suffix":"%","font":{"color":"#e8edf5","size":34,"family":"DM Mono"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#475569"},
                "bar":{"color":"#ff9933","thickness":0.3},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,35],"color":"rgba(34,197,94,0.12)"},
                    {"range":[35,65],"color":"rgba(245,158,11,0.12)"},
                    {"range":[65,100],"color":"rgba(239,68,68,0.12)"},
                ],
            }
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_gauge, use_container_width=True)
        risk_colors = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}
        st.markdown(f"""<div style="text-align:center;padding:12px;background:var(--card);border-radius:10px;border:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:22px;font-weight:700;color:{risk_colors.get(rl,'#94a3b8')};">{risk_result.get('emoji','')} {rl} Risk</div>
            <div style="font-size:12px;color:#7a8fa6;margin-top:4px;">{'Finances stable' if rl=='Low' else 'Watch your spending' if rl=='Medium' else 'Immediate action needed'}</div>
        </div>""", unsafe_allow_html=True)

    with col_emi:
        st.markdown("**💳 EMI & Loan Load**")
        emi_total = cat_breakdown[cat_breakdown["category"]=="EMI & Loans"]["abs_amount"].sum() if not cat_breakdown.empty else 0
        income_total = monthly["total_income"].mean() if not monthly.empty else 1
        emi_ratio = emi_total / income_total * 100 if income_total > 0 else 0

        fig_emi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=emi_ratio,
            title={"text":"EMI as % of Income","font":{"color":"#7a8fa6","size":12}},
            number={"suffix":"%","font":{"color":"#e8edf5","size":30,"family":"DM Mono"}},
            gauge={
                "axis":{"range":[0,60]},
                "bar":{"color":"#7c3aed"},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,30],"color":"rgba(34,197,94,0.1)"},
                    {"range":[30,40],"color":"rgba(245,158,11,0.1)"},
                    {"range":[40,60],"color":"rgba(239,68,68,0.1)"},
                ],
                "threshold":{"line":{"color":"#ef4444","width":2},"thickness":0.8,"value":40},
            }
        ))
        fig_emi.update_layout(**PLOTLY_LAYOUT, height=220)
        st.plotly_chart(fig_emi, use_container_width=True)
        emi_status = "✅ Safe" if emi_ratio < 30 else ("⚠️ Moderate" if emi_ratio < 40 else "🚨 Danger Zone")
        st.markdown(f"""<div style="text-align:center;padding:10px;background:var(--card);border-radius:8px;">
            <b>{emi_status}</b> — RBI recommends &lt;40% EMI ratio<br>
            <span style="font-size:13px;font-family:'DM Mono';">₹{emi_total:,.0f}/month in EMIs</span>
        </div>""", unsafe_allow_html=True)

    with col_detail:
        if features_df is not None and not features_df.empty and model is not None:
            st.markdown("**📅 Monthly Risk History**")
            feat_matrix = features_df[FEATURE_COLS].fillna(0)
            risk_probs = model.predict_proba(feat_matrix)[:, 1]
            emi_ratios = features_df["emi_ratio"].values if "emi_ratio" in features_df.columns else 0
            rh = pd.DataFrame({
                "month": features_df["month"].values,
                "risk_%": risk_probs * 100,
                "expense_ratio_%": features_df["expense_ratio"].values * 100,
                "emi_ratio_%": emi_ratios * 100,
            })
            fig_rh = go.Figure()
            fig_rh.add_trace(go.Bar(name="Risk %", x=rh["month"], y=rh["risk_%"],
                                     marker_color=["#ef4444" if v>65 else "#f59e0b" if v>35 else "#22c55e"
                                                   for v in rh["risk_%"]], opacity=0.8))
            fig_rh.add_trace(go.Scatter(name="Expense %", x=rh["month"], y=rh["expense_ratio_%"],
                                         line=dict(color="#ff9933",width=2), mode="lines+markers"))
            fig_rh.update_layout(**PLOTLY_LAYOUT, height=280, barmode="overlay",
                                  legend=dict(orientation="h",y=1.1),
                                  yaxis_title="%")
            st.plotly_chart(fig_rh, use_container_width=True)

    # UPI Micro spend analysis
    st.markdown("#### 📲 UPI Micro-Spend Analysis")
    micro = df[(df["amount"] < 0) & (df["abs_amount"] < 500)].copy()
    if not micro.empty:
        micro_by_cat = micro.groupby("category")["abs_amount"].agg(["sum","count"]).reset_index()
        micro_by_cat.columns = ["Category","Total (₹)","Transactions"]
        micro_by_cat = micro_by_cat.sort_values("Total (₹)", ascending=False)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_m = px.bar(micro_by_cat.head(8), x="Category", y="Total (₹)",
                           color="Total (₹)", color_continuous_scale=["#1a3a5c","#ff9933"])
            fig_m.update_layout(**PLOTLY_LAYOUT, height=250, coloraxis_showscale=False,
                                 title="Small UPI Payments by Category")
            st.plotly_chart(fig_m, use_container_width=True)
        with col_m2:
            total_micro = micro["abs_amount"].sum()
            micro_count = len(micro)
            st.markdown(f"""
            <div style="background:var(--card);border-radius:12px;padding:20px;margin-top:10px;">
                <div style="font-size:13px;color:#7a8fa6;margin-bottom:16px;text-transform:uppercase;letter-spacing:1px;">Micro-Spend Summary</div>
                <div style="display:flex;flex-direction:column;gap:12px;">
                    <div><span style="color:#7a8fa6;font-size:12px;">Total micro payments (&lt;₹500)</span><br>
                         <span style="font-size:22px;font-family:'DM Mono';font-weight:600;">₹{total_micro:,.0f}</span></div>
                    <div><span style="color:#7a8fa6;font-size:12px;">Number of transactions</span><br>
                         <span style="font-size:22px;font-family:'DM Mono';font-weight:600;">{micro_count}</span></div>
                    <div><span style="color:#7a8fa6;font-size:12px;">Avg per transaction</span><br>
                         <span style="font-size:22px;font-family:'DM Mono';font-weight:600;">₹{total_micro/micro_count:.0f}</span></div>
                </div>
                <div style="margin-top:14px;font-size:12px;color:#f59e0b;background:rgba(245,158,11,0.08);border-radius:8px;padding:10px;">
                    ⚡ These small UPI payments add up to <b>₹{total_micro:,.0f}</b> — review weekly!
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# TAB 3: INDIA BENCHMARKS
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🆚 Your Spending vs India Average")
    st.markdown("<p style='color:#7a8fa6;'>Based on NSSO/RBI urban household data. Amounts in ₹/month.</p>", unsafe_allow_html=True)

    benchmarks = benchmark_comparison(df)

    if not benchmarks:
        st.info("Not enough categorized data for benchmark comparison. Try loading more transactions.")
    else:
        col_b1, col_b2 = st.columns([2,3])

        with col_b1:
            for b in benchmarks:
                status = b["status"]
                color_map = {"high":"#ef4444","above":"#f59e0b","normal":"#22c55e","low":"#3b82f6"}
                color = color_map.get(status,"#94a3b8")
                icon = INDIA_CATEGORY_ICONS.get(b["category"],"📦")
                st.markdown(f"""
                <div class="bench-card bench-{status}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:14px;font-weight:600;">{icon} {b['category']}</span>
                        <span style="font-size:11px;padding:2px 8px;border-radius:20px;
                               background:rgba(255,255,255,0.06);color:{color};">{b['label']}</span>
                    </div>
                    <div style="display:flex;gap:20px;margin-top:8px;">
                        <div><span style="font-size:11px;color:#7a8fa6;">You</span><br>
                             <span style="font-family:'DM Mono';font-size:15px;color:{color};">₹{b['your_spend']:,.0f}</span></div>
                        <div><span style="font-size:11px;color:#7a8fa6;">India Avg</span><br>
                             <span style="font-family:'DM Mono';font-size:15px;">₹{b['avg_india']:,.0f}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_b2:
            bench_df = pd.DataFrame(benchmarks)
            if not bench_df.empty:
                fig_bench = go.Figure()
                fig_bench.add_trace(go.Bar(name="Your Spending", x=bench_df["category"],
                                           y=bench_df["your_spend"], marker_color="#ff9933", opacity=0.85))
                fig_bench.add_trace(go.Bar(name="India Average", x=bench_df["category"],
                                           y=bench_df["avg_india"], marker_color="#1a73e8", opacity=0.6))
                fig_bench.update_layout(**PLOTLY_LAYOUT, barmode="group", height=350,
                                         yaxis_title="₹/month",
                                         legend=dict(orientation="h",y=1.08),
                                         xaxis=dict(tickangle=-30))
                st.plotly_chart(fig_bench, use_container_width=True)

                # Radar chart
                st.markdown("#### 🕸️ Spending Profile Radar")
                cats = bench_df["category"].tolist()
                user_vals = bench_df["your_spend"].tolist()
                avg_vals = bench_df["avg_india"].tolist()
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=user_vals+[user_vals[0]], theta=cats+[cats[0]],
                                                     fill="toself", name="You",
                                                     line=dict(color="#ff9933"), fillcolor="rgba(255,153,51,0.15)"))
                fig_radar.add_trace(go.Scatterpolar(r=avg_vals+[avg_vals[0]], theta=cats+[cats[0]],
                                                     fill="toself", name="India Avg",
                                                     line=dict(color="#1a73e8"), fillcolor="rgba(26,115,232,0.1)"))
                fig_radar.update_layout(**PLOTLY_LAYOUT, height=350,
                                         polar=dict(bgcolor="rgba(0,0,0,0)",
                                                    radialaxis=dict(gridcolor="rgba(255,255,255,0.08)")),
                                         legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 4: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("## 💡 Personalised Financial Recommendations")
    st.markdown("<p style='color:#7a8fa6;'>Based on your actual UPI/bank transactions + Indian financial benchmarks.</p>", unsafe_allow_html=True)

    recs = generate_india_recommendations(df, risk_result.get("risk_probability",0), monthly)

    col_r1, col_r2 = st.columns([3,2])
    with col_r1:
        for rec in recs:
            st.markdown(f"""
            <div class="rec-card {rec['priority']}">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="font-size:20px;">{rec['icon']}</span>
                    <span style="font-weight:600;font-size:15px;">{rec['title']}</span>
                    <span style="margin-left:auto;font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;
                           background:{'rgba(239,68,68,0.12)' if rec['priority']=='High' else 'rgba(245,158,11,0.12)' if rec['priority']=='Medium' else 'rgba(34,197,94,0.1)'};
                           color:{'#ef4444' if rec['priority']=='High' else '#f59e0b' if rec['priority']=='Medium' else '#22c55e'};">
                        {rec['priority']}
                    </span>
                </div>
                <div style="font-size:13px;color:#7a8fa6;line-height:1.6;">{rec['detail']}</div>
            </div>""", unsafe_allow_html=True)

    with col_r2:
        st.markdown("**📊 Financial Health Score**")
        sr_val = monthly.iloc[-1]["savings_rate"] if not monthly.empty else 0
        health = min(100, max(0, sr_val*2.5 + (1-risk_result.get("risk_probability",0))*30 + 10))
        fig_h = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text":"Health Score","font":{"color":"#7a8fa6","size":13}},
            number={"suffix":"/100","font":{"color":"#e8edf5","size":28,"family":"DM Mono"}},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#ff9933"},
                "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                "steps":[
                    {"range":[0,40],"color":"rgba(239,68,68,0.1)"},
                    {"range":[40,70],"color":"rgba(245,158,11,0.1)"},
                    {"range":[70,100],"color":"rgba(34,197,94,0.1)"},
                ],
            }
        ))
        fig_h.update_layout(**PLOTLY_LAYOUT, height=230)
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("**📈 SIP Savings Projection**")
        months_f = list(range(1,13))
        avg_save = monthly["net_flow"].mean() if not monthly.empty else 2000
        current_bal = df["balance"].iloc[-1] if "balance" in df.columns else 20000
        fig_sip = go.Figure()
        for label, rate, color in [("No SIP (bank)",0.04,"#64748b"),
                                     ("SIP @ 12%",0.12,"#ff9933"),
                                     ("ELSS @ 15%",0.15,"#22c55e")]:
            vals = [current_bal * (1 + rate/12)**m + avg_save*m*0.5 for m in months_f]
            fig_sip.add_trace(go.Scatter(x=[f"+{m}m" for m in months_f], y=vals,
                                          name=label, mode="lines", line=dict(color=color,width=2)))
        fig_sip.update_layout(**PLOTLY_LAYOUT, height=220, yaxis_title="₹",
                               legend=dict(orientation="h",y=1.12,font=dict(size=10)))
        st.plotly_chart(fig_sip, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 5: AI ADVISOR
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("## 💬 AI Financial Advisor — India Edition")

    cat_json = cat_breakdown.to_dict("records") if not cat_breakdown.empty else []
    cat_str = ", ".join([f"{r['category']}: ₹{r['abs_amount']:.0f}" for r in cat_json[:6]])
    latest_m = monthly.iloc[-1] if not monthly.empty else {}

    financial_context = f"""
USER FINANCIAL PROFILE (India):
- Bank: {bank_name}
- Transactions: {len(df)} | Period: {df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}
- Monthly Income: ₹{latest_m.get('total_income',0):,.0f}
- Monthly Expenses: ₹{latest_m.get('total_expenses',0):,.0f}
- Net Savings: ₹{latest_m.get('net_flow',0):,.0f}
- Savings Rate: {latest_m.get('savings_rate',0):.1f}%
- Risk Level: {risk_result.get('risk_level','Unknown')} ({risk_result.get('risk_probability',0)*100:.0f}%)
- Top Spending: {cat_str}
- Currency: Indian Rupees (₹)
- Context: Indian financial products (SIP, PPF, NPS, ELSS, FD, UPI, NACH)
""".strip()

    # Suggested questions
    suggestions = [
        "Where am I overspending vs average Indians?",
        "How much SIP should I start with?",
        "How to reduce my UPI food spending?",
        "Am I saving enough for retirement?",
        "Should I prepay my EMI or invest?",
    ]
    st.markdown("**💡 Quick Questions:**")
    cols_s = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with cols_s[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":sug})
                with st.spinner("FinSight AI thinking..."):
                    resp = get_ai_response(sug, financial_context, st.session_state.chat_history)
                st.session_state.chat_history.append({"role":"assistant","content":resp})

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div style="text-align:center;padding:36px;color:#475569;">
            <div style="font-size:38px;margin-bottom:10px;">🤖</div>
            <div style="font-size:15px;font-weight:600;margin-bottom:6px;">FinSight India AI is ready</div>
            <div style="font-size:13px;">Ask me about SIP, tax saving, EMI strategy, UPI budgeting — anything money-related!</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-user"><div class="chat-label">You</div>{msg['content']}</div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-ai"><div class="chat-label">🇮🇳 FinSight AI</div>{msg['content']}</div>""",
                            unsafe_allow_html=True)

    col_i, col_s, col_c = st.columns([6,1,1])
    with col_i:
        user_input = st.text_input("Ask anything about your finances...", key="chat_in",
                                    label_visibility="collapsed",
                                    placeholder="e.g. How can I save ₹5,000 more this month?")
    with col_s:
        send = st.button("Send", use_container_width=True)
    with col_c:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            resp = get_ai_response(user_input, financial_context, st.session_state.chat_history)
        st.session_state.chat_history.append({"role":"assistant","content":resp})
        st.rerun()

    with st.expander("🔍 View context sent to AI"):
        st.code(financial_context, language="text")
