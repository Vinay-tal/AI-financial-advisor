"""
Data processing utilities for AI Financial Behavior Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime


CATEGORY_COLORS = {
    "Income": "#22c55e",
    "Housing": "#6366f1",
    "Groceries": "#f59e0b",
    "Dining": "#ef4444",
    "Transport": "#3b82f6",
    "Entertainment": "#a855f7",
    "Shopping": "#ec4899",
    "Health": "#10b981",
    "Utilities": "#64748b",
    "Savings": "#06b6d4",
    "Other": "#94a3b8",
}

CATEGORY_ICONS = {
    "Income": "💰",
    "Housing": "🏠",
    "Groceries": "🛒",
    "Dining": "🍽️",
    "Transport": "🚗",
    "Entertainment": "🎬",
    "Shopping": "🛍️",
    "Health": "💊",
    "Utilities": "⚡",
    "Savings": "🏦",
    "Other": "📦",
}


def load_and_clean(file) -> pd.DataFrame:
    """Load CSV and standardize columns."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize column names
    col_map = {}
    for col in df.columns:
        if "date" in col:
            col_map[col] = "date"
        elif "desc" in col or "narration" in col or "particular" in col:
            col_map[col] = "description"
        elif "amount" in col or "amt" in col:
            col_map[col] = "amount"
        elif "type" in col or "dr" in col or "cr" in col:
            col_map[col] = "type"
        elif "categ" in col:
            col_map[col] = "category"
        elif "balance" in col or "bal" in col:
            col_map[col] = "balance"
    df.rename(columns=col_map, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    if "category" not in df.columns:
        df["category"] = df.apply(lambda r: auto_categorize(r.get("description", "")), axis=1)

    if "type" not in df.columns:
        df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")

    df["abs_amount"] = df["amount"].abs()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_month"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()

    return df


def auto_categorize(description: str) -> str:
    desc = description.lower()
    rules = {
        "Income": ["salary", "income", "deposit", "freelance", "payment received", "credit"],
        "Housing": ["rent", "mortgage", "property", "landlord"],
        "Groceries": ["grocery", "supermarket", "food mart", "walmart", "target", "costco"],
        "Dining": ["restaurant", "cafe", "coffee", "starbucks", "mcdonald", "pizza", "food", "bar", "drinks", "fast food"],
        "Transport": ["uber", "lyft", "gas", "fuel", "petrol", "metro", "bus", "train", "parking", "toll"],
        "Entertainment": ["netflix", "spotify", "amazon prime", "hulu", "movie", "concert", "streaming", "gaming"],
        "Shopping": ["amazon", "ebay", "shopping", "store", "clothing", "electronics", "purchase"],
        "Health": ["pharmacy", "doctor", "hospital", "gym", "fitness", "medical", "dental"],
        "Utilities": ["electricity", "water", "gas bill", "internet", "phone", "cable", "utility"],
        "Savings": ["transfer", "savings", "investment", "mutual fund"],
    }
    for category, keywords in rules.items():
        if any(k in desc for k in keywords):
            return category
    return "Other"


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("month").agg(
        total_income=("amount", lambda x: x[x > 0].sum()),
        total_expenses=("amount", lambda x: x[x < 0].abs().sum()),
        net_flow=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_transaction=("abs_amount", "mean"),
    ).reset_index()
    monthly["savings_rate"] = (
        (monthly["total_income"] - monthly["total_expenses"]) / monthly["total_income"].replace(0, np.nan) * 100
    ).fillna(0)
    return monthly


def get_category_breakdown(df: pd.DataFrame, month: str = None) -> pd.DataFrame:
    data = df[df["amount"] < 0].copy()
    if month:
        data = data[data["month"] == month]
    breakdown = (
        data.groupby("category")["abs_amount"]
        .sum()
        .reset_index()
        .sort_values("abs_amount", ascending=False)
    )
    breakdown["percentage"] = breakdown["abs_amount"] / breakdown["abs_amount"].sum() * 100
    return breakdown


def get_spending_trends(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    trend = expenses.groupby(["month", "category"])["abs_amount"].sum().reset_index()
    return trend


def compute_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features per month for risk model."""
    records = []
    for month, grp in df.groupby("month"):
        income = grp[grp["amount"] > 0]["amount"].sum()
        expenses = grp[grp["amount"] < 0]["amount"].abs().sum()
        balance_end = grp["balance"].iloc[-1] if "balance" in grp.columns else (income - expenses)
        balance_start = grp["balance"].iloc[0] if "balance" in grp.columns else income
        dining = grp[grp["category"] == "Dining"]["abs_amount"].sum()
        entertainment = grp[grp["category"] == "Entertainment"]["abs_amount"].sum()
        shopping = grp[grp["category"] == "Shopping"]["abs_amount"].sum()
        num_transactions = len(grp)
        avg_daily_spend = expenses / 30
        expense_ratio = expenses / income if income > 0 else 1.0
        discretionary_ratio = (dining + entertainment + shopping) / expenses if expenses > 0 else 0
        balance_drop = (balance_start - balance_end) / balance_start if balance_start > 0 else 0

        # Risk label: 1 if expenses > 90% of income OR balance dropped > 30%
        risk = 1 if (expense_ratio > 0.90 or balance_drop > 0.30) else 0

        records.append({
            "month": month,
            "income": income,
            "expenses": expenses,
            "balance_end": balance_end,
            "dining_spend": dining,
            "entertainment_spend": entertainment,
            "shopping_spend": shopping,
            "num_transactions": num_transactions,
            "avg_daily_spend": avg_daily_spend,
            "expense_ratio": expense_ratio,
            "discretionary_ratio": discretionary_ratio,
            "balance_drop": balance_drop,
            "risk_label": risk,
        })
    return pd.DataFrame(records)


def generate_recommendations(df: pd.DataFrame, risk_score: float) -> list:
    """Generate rule-based financial recommendations."""
    recs = []
    cat = get_category_breakdown(df)
    cat_dict = dict(zip(cat["category"], cat["abs_amount"]))
    monthly = get_monthly_summary(df)
    latest = monthly.iloc[-1] if not monthly.empty else None

    if latest is not None:
        savings_rate = latest["savings_rate"]
        if savings_rate < 10:
            recs.append({"icon": "🚨", "priority": "High", "title": "Critically Low Savings Rate",
                         "detail": f"You're saving only {savings_rate:.1f}% of income. Aim for 20%+. Try automating a fixed transfer to savings on payday."})
        elif savings_rate < 20:
            recs.append({"icon": "⚠️", "priority": "Medium", "title": "Below-Target Savings",
                         "detail": f"Savings rate is {savings_rate:.1f}%. Small cuts in discretionary spending can push you to the 20% goal."})

    dining = cat_dict.get("Dining", 0)
    if dining > 300:
        recs.append({"icon": "🍽️", "priority": "Medium", "title": "High Dining Expenses",
                     "detail": f"You spent ${dining:.0f} on dining. Cooking 3 meals at home per week could save ~$150/month."})

    shopping = cat_dict.get("Shopping", 0)
    if shopping > 400:
        recs.append({"icon": "🛍️", "priority": "Medium", "title": "Elevated Shopping Spend",
                     "detail": f"${shopping:.0f} on shopping this period. Try a 48-hour rule: wait 2 days before non-essential purchases."})

    entertainment = cat_dict.get("Entertainment", 0)
    if entertainment > 200:
        recs.append({"icon": "🎬", "priority": "Low", "title": "Entertainment Costs",
                     "detail": f"${entertainment:.0f} on entertainment. Review subscriptions — cancel unused ones. Even $30/month adds up to $360/year."})

    if risk_score > 0.6:
        recs.append({"icon": "🔴", "priority": "High", "title": "High Risk of Running Short",
                     "detail": "Based on your spending trajectory, you may run short before month end. Cut discretionary spending now and prioritize essentials."})

    if not recs:
        recs.append({"icon": "✅", "emoji": "✅", "priority": "Low", "title": "Finances Look Healthy",
                     "detail": "You're on track! Keep maintaining this discipline. Consider investing surplus savings in index funds or ETFs."})

    # Ensure all recs have emoji key (alias of icon)
    for r in recs:
        if "emoji" not in r:
            r["emoji"] = r.get("icon", "💡")

    return recs
