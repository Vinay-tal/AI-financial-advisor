"""
AI Financial Behavior Engine
A production-grade fintech AI system built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_processor import (
    load_and_clean, get_monthly_summary, get_category_breakdown,
    get_spending_trends, compute_risk_features, generate_recommendations,
    CATEGORY_COLORS, CATEGORY_ICONS,
)
from utils.risk_model import train_model, load_model, predict_risk, get_feature_importance
from utils.ai_advisor import build_financial_context, get_ai_response

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Financial Behavior Engine",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent-green: #00ff88;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-red: #ef4444;
    --accent-orange: #f59e0b;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --border: rgba(255,255,255,0.07);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-primary) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(59,130,246,0.3);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}
.metric-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: 700; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }
.metric-delta { font-size: 13px; margin-top: 6px; }
.delta-pos { color: var(--accent-green); }
.delta-neg { color: var(--accent-red); }

/* Risk badge */
.risk-card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid var(--border);
    text-align: center;
}
.risk-score {
    font-size: 64px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.risk-low { color: #00ff88; }
.risk-med { color: #f59e0b; }
.risk-high { color: #ef4444; }

/* Recommendation cards */
.rec-card {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.rec-card.high { border-left-color: var(--accent-red); }
.rec-card.medium { border-left-color: var(--accent-orange); }
.rec-card.low { border-left-color: var(--accent-green); }
.rec-title { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
.rec-detail { font-size: 13px; color: var(--text-muted); line-height: 1.5; }

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2e4a);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-left: 15%;
    font-size: 14px;
    border: 1px solid rgba(59,130,246,0.2);
}
.chat-ai {
    background: var(--bg-card2);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    margin-right: 15%;
    font-size: 14px;
    border: 1px solid var(--border);
}
.chat-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    color: var(--text-muted);
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 700;
    margin: 32px 0 20px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed rgba(59,130,246,0.4);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    background: rgba(59,130,246,0.03);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stToolbar"] {display: none;}

/* Plotly chart background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Input styling */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-muted) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(59,130,246,0.4) !important;
    border-radius: 12px !important;
}

/* Progress / spinner */
.stProgress > div > div { background: var(--accent-blue) !important; }

/* DataFrame */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "risk_result" not in st.session_state:
    st.session_state.risk_result = {}


# ─────────────────────────────────────────────────────────────
# HELPER: PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 10px 0;">
        <div style="font-size:28px; font-weight:800; background: linear-gradient(135deg, #3b82f6, #00ff88); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">💹 FinSight</div>
        <div style="font-size:12px; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">AI Financial Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📂 Upload Transaction Data**")
    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    default_file = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")
    use_demo = st.checkbox("Use demo dataset", value=(uploaded_file is None))

    if st.button("🔄 Load & Analyze", use_container_width=True):
        with st.spinner("Processing transactions..."):
            try:
                src = default_file if use_demo else uploaded_file
                df = load_and_clean(src)
                st.session_state.df = df

                # Train ML model
                features_df = compute_risk_features(df)
                model = train_model(features_df)
                st.session_state.model = model
                st.session_state.features_df = features_df

                # Compute current month risk
                if not features_df.empty:
                    latest_feat = features_df.iloc[-1].to_dict()
                    st.session_state.risk_result = predict_risk(model, latest_feat)

                st.success(f"✅ Loaded {len(df):,} transactions")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Groq API key input
    st.markdown("**🤖 AI Advisor**")
    groq_key = st.text_input("Groq API Key (optional)", type="password",
                              placeholder="gsk_...",
                              help="Get free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        import utils.ai_advisor as ai_mod
        ai_mod.GROQ_API_KEY = groq_key

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#475569; line-height:1.8;">
    <b style="color:#64748b;">CSV FORMAT:</b><br>
    date, description, amount,<br>
    type, category, balance<br><br>
    <b style="color:#64748b;">SUPPORTED:</b><br>
    • Bank statements<br>
    • Credit card exports<br>
    • Any transaction CSV
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    # Landing screen
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size:72px; margin-bottom:24px;">💹</div>
        <h1 style="font-size:48px; font-weight:800; background: linear-gradient(135deg, #3b82f6, #00ff88, #8b5cf6);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:16px;">
            FinSight AI
        </h1>
        <p style="font-size:20px; color:#64748b; max-width:600px; margin:0 auto 40px auto; line-height:1.6;">
            Intelligent financial analysis powered by machine learning.<br>
            Upload your transactions. Understand your money.
        </p>
        <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin-bottom:60px;">
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:20px 28px;">
                <div style="font-size:28px; margin-bottom:8px;">📊</div>
                <div style="font-weight:600; margin-bottom:4px;">Smart Analytics</div>
                <div style="font-size:13px; color:#64748b;">Category breakdowns & trends</div>
            </div>
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:20px 28px;">
                <div style="font-size:28px; margin-bottom:8px;">🤖</div>
                <div style="font-weight:600; margin-bottom:4px;">Risk Prediction</div>
                <div style="font-size:13px; color:#64748b;">ML model trained on your data</div>
            </div>
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:20px 28px;">
                <div style="font-size:28px; margin-bottom:8px;">💬</div>
                <div style="font-weight:600; margin-bottom:4px;">AI Advisor</div>
                <div style="font-size:13px; color:#64748b;">Chat with your financial AI</div>
            </div>
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:20px 28px;">
                <div style="font-size:28px; margin-bottom:8px;">💡</div>
                <div style="font-weight:600; margin-bottom:4px;">Recommendations</div>
                <div style="font-size:13px; color:#64748b;">Actionable savings strategies</div>
            </div>
        </div>
        <p style="color:#3b82f6; font-weight:600;">← Upload your CSV or click "Use demo dataset" in the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# DATA IS LOADED — SHOW FULL DASHBOARD
# ─────────────────────────────────────────────────────────────
df = st.session_state.df
model = st.session_state.model
risk_result = st.session_state.risk_result
monthly = get_monthly_summary(df)
cat_breakdown = get_category_breakdown(df)
features_df = st.session_state.get("features_df", compute_risk_features(df))

# Top bar
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; padding:0 0 24px 0; border-bottom:1px solid rgba(255,255,255,0.07); margin-bottom:28px;">
    <div>
        <div style="font-size:26px; font-weight:800;">💹 FinSight AI Dashboard</div>
        <div style="font-size:13px; color:#64748b; margin-top:2px;">Financial Behavior Engine — Real-time Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Risk Analysis", "💡 Recommendations", "💬 AI Advisor"])


# ═══════════════════════════════════════════════════════
# TAB 1: OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════
with tab1:
    latest = monthly.iloc[-1] if not monthly.empty else {}

    # KPI Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        income = latest.get("total_income", 0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Income</div>
            <div class="metric-value">${income:,.0f}</div>
            <div class="metric-delta delta-pos">💰 This month</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        expenses = latest.get("total_expenses", 0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Expenses</div>
            <div class="metric-value">${expenses:,.0f}</div>
            <div class="metric-delta delta-neg">💸 Total spend</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        net = latest.get("net_flow", 0)
        delta_class = "delta-pos" if net >= 0 else "delta-neg"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Net Cash Flow</div>
            <div class="metric-value">${net:,.0f}</div>
            <div class="metric-delta {delta_class}">{'▲ Surplus' if net >= 0 else '▼ Deficit'}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sr = latest.get("savings_rate", 0)
        sr_class = "delta-pos" if sr >= 20 else ("delta-neg" if sr < 10 else "")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Savings Rate</div>
            <div class="metric-value">{sr:.1f}%</div>
            <div class="metric-delta {sr_class}">{'✅ Great' if sr >= 20 else ('⚠️ Low' if sr < 10 else '📈 Okay')}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        txn_count = latest.get("transaction_count", len(df))
        risk_level = risk_result.get("risk_level", "—")
        risk_emoji = risk_result.get("emoji", "⚪")
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk_emoji} {risk_level}</div>
            <div class="metric-delta">{txn_count} transactions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">📈 Income vs Expenses Over Time</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Income", x=monthly["month"], y=monthly["total_income"],
                             marker_color="#00ff88", opacity=0.85))
        fig.add_trace(go.Bar(name="Expenses", x=monthly["month"], y=monthly["total_expenses"],
                             marker_color="#ef4444", opacity=0.85))
        fig.add_trace(go.Scatter(name="Net Flow", x=monthly["month"], y=monthly["net_flow"],
                                 mode="lines+markers", line=dict(color="#3b82f6", width=2.5),
                                 marker=dict(size=7)))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=320,
                          legend=dict(orientation="h", y=1.1),
                          title=dict(text="Monthly Financial Overview", font=dict(size=14, color="#f8fafc")))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🏷️ Spending by Category</div>', unsafe_allow_html=True)
        colors = [CATEGORY_COLORS.get(c, "#94a3b8") for c in cat_breakdown["category"]]
        fig2 = go.Figure(go.Pie(
            labels=cat_breakdown["category"],
            values=cat_breakdown["abs_amount"],
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textinfo="percent+label",
            textfont=dict(size=12),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           title=dict(text="Category Distribution", font=dict(size=14, color="#f8fafc")),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 3: Trends + Category Table
    col3a, col3b = st.columns([3, 2])

    with col3a:
        st.markdown('<div class="section-header">📉 Spending Trends by Category</div>', unsafe_allow_html=True)
        trend = get_spending_trends(df)
        top_cats = cat_breakdown.head(6)["category"].tolist()
        trend_filtered = trend[trend["category"].isin(top_cats)]
        fig3 = px.line(trend_filtered, x="month", y="abs_amount", color="category",
                       color_discrete_map=CATEGORY_COLORS, markers=True)
        fig3.update_traces(line=dict(width=2.5))
        fig3.update_layout(**PLOTLY_LAYOUT, height=300,
                           yaxis_title="Amount ($)", xaxis_title="",
                           legend=dict(orientation="h", y=1.1, font=dict(size=11)))
        st.plotly_chart(fig3, use_container_width=True)

    with col3b:
        st.markdown('<div class="section-header">📋 Category Breakdown</div>', unsafe_allow_html=True)
        for _, row in cat_breakdown.iterrows():
            icon = CATEGORY_ICONS.get(row["category"], "📦")
            color = CATEGORY_COLORS.get(row["category"], "#94a3b8")
            pct = row["percentage"]
            bar_width = int(pct)
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:13px;">{icon} {row['category']}</span>
                    <span style="font-size:13px; font-family:'JetBrains Mono'; color:{color};">${row['abs_amount']:,.0f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:6px;">
                    <div style="width:{bar_width}%; height:100%; border-radius:4px; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Row 4: Recent transactions
    st.markdown('<div class="section-header">🧾 Recent Transactions</div>', unsafe_allow_html=True)
    recent_cols = ["date", "description", "category", "amount"] + (["balance"] if "balance" in df.columns else [])
    recent = df.tail(20)[recent_cols].copy()
    recent["date"] = recent["date"].dt.strftime("%b %d, %Y")
    recent["amount"] = recent["amount"].apply(lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}")
    st.dataframe(recent.iloc[::-1], use_container_width=True, height=320,
                 column_config={
                     "date": "Date", "description": "Description",
                     "category": "Category", "amount": "Amount", "balance": "Balance"
                 })


# ═══════════════════════════════════════════════════════
# TAB 2: RISK ANALYSIS
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔮 Financial Risk Prediction")
    st.markdown("<p style='color:#64748b;'>Machine learning model predicts if you'll run out of money before month end.</p>", unsafe_allow_html=True)

    col_risk, col_gauge, col_feat = st.columns([1, 2, 2])

    with col_risk:
        rp = risk_result.get("risk_probability", 0)
        rl = risk_result.get("risk_level", "Unknown")
        emoji = risk_result.get("emoji", "⚪")
        risk_class = {"Low": "risk-low", "Medium": "risk-med", "High": "risk-high"}.get(rl, "")
        st.markdown(f"""
        <div class="risk-card">
            <div style="font-size:13px; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;">Current Risk</div>
            <div class="risk-score {risk_class}">{rp*100:.0f}<span style="font-size:28px;">%</span></div>
            <div style="font-size:22px; margin:16px 0;">{emoji} {rl} Risk</div>
            <div style="font-size:13px; color:#64748b; line-height:1.6;">
                {'Finances are on track. Keep maintaining current discipline.' if rl == 'Low' else
                 'Some warning signs detected. Consider reducing discretionary spending.' if rl == 'Medium' else
                 'High probability of running short. Immediate action recommended.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rp * 100,
            title={"text": "Risk Score", "font": {"color": "#94a3b8", "size": 14}},
            number={"suffix": "%", "font": {"color": "#f8fafc", "size": 36, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#3b82f6", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(0,255,136,0.15)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.15)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 2}, "thickness": 0.8, "value": rp * 100},
            }
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_feat:
        if model is not None:
            st.markdown("**🧠 Feature Importance**")
            feat_imp = get_feature_importance(model)
            feat_imp["feature"] = feat_imp["feature"].str.replace("_", " ").str.title()
            fig_feat = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                              color="importance", color_continuous_scale=["#1e3a5f", "#3b82f6", "#00ff88"])
            fig_feat.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False,
                                   xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_feat, use_container_width=True)

    # Monthly risk history
    st.markdown('<div class="section-header">📅 Monthly Risk History</div>', unsafe_allow_html=True)
    if not features_df.empty:
        risk_preds = []
        for _, row in features_df.iterrows():
            r = predict_risk(model, row.to_dict())
            risk_preds.append({"month": row["month"], "risk_%": r["risk_probability"] * 100,
                                "level": r["risk_level"], "expense_ratio": row["expense_ratio"] * 100,
                                "savings_rate": (1 - row["expense_ratio"]) * 100})

        risk_hist = pd.DataFrame(risk_preds)
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist.add_trace(go.Bar(name="Risk %", x=risk_hist["month"], y=risk_hist["risk_%"],
                                  marker_color=["#ef4444" if r > 65 else "#f59e0b" if r > 35 else "#00ff88"
                                                for r in risk_hist["risk_%"]],
                                  opacity=0.8), secondary_y=False)
        fig_hist.add_trace(go.Scatter(name="Expense Ratio %", x=risk_hist["month"],
                                      y=risk_hist["expense_ratio"],
                                      line=dict(color="#8b5cf6", width=2), mode="lines+markers"),
                           secondary_y=True)
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=300,
                               title=dict(text="Risk & Expense Ratio by Month", font=dict(size=14, color="#f8fafc")),
                               legend=dict(orientation="h", y=1.1))
        fig_hist.update_yaxes(title_text="Risk Score (%)", secondary_y=False,
                              gridcolor="rgba(255,255,255,0.05)")
        fig_hist.update_yaxes(title_text="Expense Ratio (%)", secondary_y=True,
                              gridcolor="rgba(255,255,255,0.02)")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Feature table
        st.markdown('<div class="section-header">📊 Monthly Financial Metrics</div>', unsafe_allow_html=True)
        display_df = features_df[["month", "income", "expenses", "expense_ratio",
                                   "discretionary_ratio", "avg_daily_spend"]].copy()
        display_df.columns = ["Month", "Income ($)", "Expenses ($)", "Expense Ratio", "Discretionary Ratio", "Avg Daily Spend ($)"]
        display_df["Expense Ratio"] = display_df["Expense Ratio"].map("{:.1%}".format)
        display_df["Discretionary Ratio"] = display_df["Discretionary Ratio"].map("{:.1%}".format)
        display_df["Income ($)"] = display_df["Income ($)"].map("${:,.2f}".format)
        display_df["Expenses ($)"] = display_df["Expenses ($)"].map("${:,.2f}".format)
        display_df["Avg Daily Spend ($)"] = display_df["Avg Daily Spend ($)"].map("${:,.2f}".format)
        st.dataframe(display_df, use_container_width=True, height=200)


# ═══════════════════════════════════════════════════════
# TAB 3: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("## 💡 Actionable Recommendations")
    st.markdown("<p style='color:#64748b;'>Personalized advice based on your spending patterns and risk profile.</p>", unsafe_allow_html=True)

    recs = generate_recommendations(df, risk_result.get("risk_probability", 0))

    col_recs, col_summary = st.columns([3, 2])

    with col_recs:
        for rec in recs:
            priority_class = rec["priority"].lower()
            st.markdown(f"""
            <div class="rec-card {priority_class}">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <span style="font-size:20px;">{rec['icon']}</span>
                    <span class="rec-title">{rec['title']}</span>
                    <span style="margin-left:auto; font-size:10px; font-weight:700; 
                                 padding:2px 8px; border-radius:20px; 
                                 background:{'rgba(239,68,68,0.15)' if rec['priority']=='High' else 'rgba(245,158,11,0.15)' if rec['priority']=='Medium' else 'rgba(0,255,136,0.1)'};
                                 color:{'#ef4444' if rec['priority']=='High' else '#f59e0b' if rec['priority']=='Medium' else '#00ff88'};">
                        {rec['priority']}
                    </span>
                </div>
                <div class="rec-detail">{rec['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_summary:
        st.markdown("**📊 Spending Health Score**")
        sr = monthly.iloc[-1]["savings_rate"] if not monthly.empty else 0
        health_score = min(100, max(0, sr * 3 + (1 - risk_result.get("risk_probability", 0)) * 40))
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            title={"text": "Financial Health", "font": {"color": "#94a3b8", "size": 13}},
            delta={"reference": 70, "relative": False,
                   "increasing": {"color": "#00ff88"}, "decreasing": {"color": "#ef4444"}},
            number={"suffix": "/100", "font": {"color": "#f8fafc", "size": 30, "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.12)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                    {"range": [70, 100], "color": "rgba(0,255,136,0.12)"},
                ],
            }
        ))
        fig_health.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_health, use_container_width=True)

        st.markdown("**🎯 Quick Wins**")
        dining = cat_breakdown[cat_breakdown["category"] == "Dining"]["abs_amount"].sum() if not cat_breakdown.empty else 0
        shopping = cat_breakdown[cat_breakdown["category"] == "Shopping"]["abs_amount"].sum() if not cat_breakdown.empty else 0
        entertainment = cat_breakdown[cat_breakdown["category"] == "Entertainment"]["abs_amount"].sum() if not cat_breakdown.empty else 0
        
        for label, current, target_pct, icon in [
            ("Dining", dining, 0.7, "🍽️"),
            ("Shopping", shopping, 0.7, "🛍️"),
            ("Entertainment", entertainment, 0.8, "🎬"),
        ]:
            potential_save = current * (1 - target_pct)
            if potential_save > 20:
                st.markdown(f"""
                <div style="background:var(--bg-card); border-radius:10px; padding:12px 16px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.06);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:13px;">{icon} Cut {label} by 30%</span>
                        <span style="color:#00ff88; font-weight:700; font-family:'JetBrains Mono'; font-size:13px;">Save ${potential_save:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Savings projection
    st.markdown('<div class="section-header">📈 Savings Projection</div>', unsafe_allow_html=True)
    if not monthly.empty:
        avg_save = monthly["net_flow"].mean()
        months_future = list(range(1, 13))
        projections = {"Conservative (50%)": [], "Current Pace": [], "Optimized (+20%)": []}
        current_bal = df["balance"].iloc[-1] if "balance" in df.columns else 5000
        for m in months_future:
            projections["Conservative (50%)"].append(current_bal + avg_save * 0.5 * m)
            projections["Current Pace"].append(current_bal + avg_save * m)
            projections["Optimized (+20%)"].append(current_bal + avg_save * 1.2 * m)

        fig_proj = go.Figure()
        colors_proj = {"Conservative (50%)": "#f59e0b", "Current Pace": "#3b82f6", "Optimized (+20%)": "#00ff88"}
        for key, vals in projections.items():
            fig_proj.add_trace(go.Scatter(
                x=[f"Month +{m}" for m in months_future], y=vals,
                name=key, mode="lines+markers",
                line=dict(color=colors_proj[key], width=2.5),
                fill="tonexty" if key == "Optimized (+20%)" else None,
                fillcolor="rgba(0,255,136,0.05)",
            ))
        fig_proj.update_layout(**PLOTLY_LAYOUT, height=300,
                               title=dict(text="12-Month Balance Projection", font=dict(size=14, color="#f8fafc")),
                               yaxis_title="Projected Balance ($)",
                               legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_proj, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 4: AI ADVISOR CHAT
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("## 💬 AI Financial Advisor")
    st.markdown("<p style='color:#64748b;'>Chat with your personal AI financial advisor. Ask anything about your money.</p>", unsafe_allow_html=True)

    # Build context
    financial_context = build_financial_context(
        df, monthly, risk_result, cat_breakdown
    )

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = [
        "How can I save more money this month?",
        "What's my biggest spending problem?",
        "Am I at risk of running out of money?",
        "How should I invest my savings?",
        "How can I reduce my dining expenses?",
    ]
    cols_sug = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with cols_sug[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": sug})
                with st.spinner("FinSight AI is thinking..."):
                    response = get_ai_response(sug, financial_context, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:40px; color:#475569;">
                <div style="font-size:40px; margin-bottom:12px;">🤖</div>
                <div style="font-size:16px; font-weight:600; margin-bottom:8px;">FinSight AI is ready</div>
                <div style="font-size:13px;">Ask me about your spending, savings, risk, or any financial topic.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-user">
                        <div class="chat-label">You</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-ai">
                        <div class="chat-label">💹 FinSight AI</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input
    col_input, col_btn, col_clear = st.columns([6, 1, 1])
    with col_input:
        user_input = st.text_input("Ask your financial advisor...", key="chat_input",
                                    placeholder="e.g. How can I save $500 this month?",
                                    label_visibility="collapsed")
    with col_btn:
        send = st.button("Send 📤", use_container_width=True)
    with col_clear:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("FinSight AI is thinking..."):
            response = get_ai_response(user_input, financial_context, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Context preview (expandable)
    with st.expander("🔍 View financial context sent to AI", expanded=False):
        st.code(financial_context, language="text")
