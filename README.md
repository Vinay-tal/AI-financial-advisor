# 💹 AI Financial Behavior Engine — FinSight AI

A production-quality fintech AI system that analyzes transaction data, predicts financial risk, and delivers actionable advice via an LLM chat interface.

---

## 🚀 Quick Start

### 1. Clone / Extract the project
```bash
cd ai_financial_engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add Groq API Key for real AI responses
Get a **free** API key at [console.groq.com](https://console.groq.com)

Edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

---

## 📁 Project Structure

```
ai_financial_engine/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # API keys (add Groq key here)
├── data/
│   └── transactions.csv      # Sample dataset (or upload your own)
├── models/
│   └── risk_model.joblib     # Auto-generated ML model
└── utils/
    ├── data_processor.py     # Data loading, cleaning, feature engineering
    ├── risk_model.py         # Random Forest risk prediction model
    └── ai_advisor.py         # Groq LLM integration + fallback responses
```

---

## 📊 Features

| Feature | Description |
|---|---|
| **Transaction Analytics** | Auto-categorizes spending, monthly summaries, trends |
| **Spending Dashboard** | Interactive Plotly charts: income vs expenses, pie charts, bar charts |
| **Risk Prediction** | Random Forest ML model predicts if user will run short before month end |
| **AI Advisor Chat** | Groq-powered LLM (llama3-70b-8192) with financial context |
| **Recommendations** | Prioritized actionable tips based on real spending data |
| **Savings Projections** | 12-month balance forecast at 3 scenarios |

---

## 📂 CSV Format

Your CSV should have these columns (any order, flexible naming):

| Column | Required | Notes |
|---|---|---|
| `date` | ✅ | Any standard date format |
| `description` | ✅ | Transaction description |
| `amount` | ✅ | Positive = income, Negative = expense |
| `category` | Optional | Auto-detected if missing |
| `type` | Optional | credit/debit |
| `balance` | Optional | Running balance |

**Compatible with:** Bank statement exports, credit card CSV downloads, any fintech CSV.

---

## 🤖 AI Advisor

- **With Groq API Key**: Uses `llama3-70b-8192` — fast, intelligent, context-aware responses
- **Without API Key**: Uses built-in rule-based responses (still useful!)

---

## 🧠 ML Risk Model

- **Algorithm**: Random Forest Classifier with StandardScaler
- **Features**: expense ratio, discretionary spend ratio, avg daily spend, balance drop rate, category-level spend
- **Training**: Automatically trains on your data + synthetic augmentation for small datasets
- **Output**: Risk probability (0-100%) + risk level (Low/Medium/High)

---

## 🎨 Tech Stack

- **Frontend**: Streamlit + custom dark theme CSS
- **Charts**: Plotly (interactive)
- **ML**: scikit-learn (Random Forest)
- **Data**: pandas + numpy
- **LLM**: Groq API (llama3-70b-8192)
- **Persistence**: joblib model saving
