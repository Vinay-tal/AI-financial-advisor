"""
India-specific financial analytics — categories, benchmarks, recommendations
"""

import pandas as pd
import numpy as np


# ── Indian Category Config ────────────────────────────────────

INDIA_CATEGORY_COLORS = {
    "Income":             "#22c55e",
    "Food & Dining":      "#f97316",
    "Groceries":          "#eab308",
    "Transport":          "#3b82f6",
    "Travel":             "#06b6d4",
    "Shopping":           "#ec4899",
    "Recharge & Bills":   "#8b5cf6",
    "Utilities":          "#64748b",
    "Health":             "#10b981",
    "Entertainment":      "#a855f7",
    "Education":          "#0ea5e9",
    "Investment":         "#84cc16",
    "Insurance":          "#14b8a6",
    "EMI & Loans":        "#ef4444",
    "Housing & Rent":     "#6366f1",
    "Savings & Transfer": "#06b6d4",
    "UPI Transfer":       "#94a3b8",
    "Cash":               "#d97706",
    "Others":             "#475569",
}

INDIA_CATEGORY_ICONS = {
    "Income":             "💰",
    "Food & Dining":      "🍱",
    "Groceries":          "🛒",
    "Transport":          "🚗",
    "Travel":             "✈️",
    "Shopping":           "🛍️",
    "Recharge & Bills":   "📱",
    "Utilities":          "⚡",
    "Health":             "💊",
    "Entertainment":      "🎬",
    "Education":          "📚",
    "Investment":         "📈",
    "Insurance":          "🛡️",
    "EMI & Loans":        "🏦",
    "Housing & Rent":     "🏠",
    "Savings & Transfer": "💸",
    "UPI Transfer":       "📲",
    "Cash":               "💵",
    "Others":             "📦",
}

# ── Indian Household Spending Benchmarks (monthly, ₹) ─────────
# Based on NSSO / RBI / NHB data for urban India
INDIA_BENCHMARKS = {
    "Food & Dining":    {"low": 2000,  "avg": 4500,  "high": 9000},
    "Groceries":        {"low": 3000,  "avg": 6000,  "high": 12000},
    "Transport":        {"low": 1000,  "avg": 3000,  "high": 7000},
    "Recharge & Bills": {"low": 300,   "avg": 700,   "high": 1500},
    "Utilities":        {"low": 500,   "avg": 1500,  "high": 4000},
    "Health":           {"low": 500,   "avg": 2000,  "high": 6000},
    "Entertainment":    {"low": 200,   "avg": 800,   "high": 3000},
    "Shopping":         {"low": 1000,  "avg": 3000,  "high": 10000},
    "Education":        {"low": 500,   "avg": 3000,  "high": 15000},
    "Housing & Rent":   {"low": 5000,  "avg": 15000, "high": 40000},
    "EMI & Loans":      {"low": 0,     "avg": 5000,  "high": 25000},
}


def get_india_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("month").agg(
        total_income=("amount", lambda x: x[x > 0].sum()),
        total_expenses=("amount", lambda x: x[x < 0].abs().sum()),
        net_flow=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_transaction=("abs_amount", "mean"),
        upi_count=("source", lambda x: (x == "SMS").sum() if "source" in df.columns else 0),
    ).reset_index()
    monthly["savings_rate"] = (
        (monthly["total_income"] - monthly["total_expenses"])
        / monthly["total_income"].replace(0, np.nan) * 100
    ).fillna(0).clip(lower=-999, upper=100)
    return monthly


def get_india_category_breakdown(df: pd.DataFrame, month: str = None) -> pd.DataFrame:
    data = df[df["amount"] < 0].copy()
    if month:
        data = data[data["month"] == month]
    breakdown = (
        data.groupby("category")["abs_amount"]
        .sum()
        .reset_index()
        .sort_values("abs_amount", ascending=False)
    )
    total = breakdown["abs_amount"].sum()
    breakdown["percentage"] = (breakdown["abs_amount"] / total * 100).round(1) if total > 0 else 0
    return breakdown


def get_india_spending_trends(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    return expenses.groupby(["month", "category"])["abs_amount"].sum().reset_index()


def benchmark_comparison(df: pd.DataFrame) -> list:
    """Compare user spending vs Indian household benchmarks."""
    cat = get_india_category_breakdown(df)
    results = []
    for _, row in cat.iterrows():
        cat_name = row["category"]
        if cat_name not in INDIA_BENCHMARKS:
            continue
        bench = INDIA_BENCHMARKS[cat_name]
        user_spend = row["abs_amount"]
        avg = bench["avg"]
        ratio = user_spend / avg if avg > 0 else 0
        if ratio > 1.5:
            status = "high"
            label = f"{ratio:.1f}x above avg"
        elif ratio > 1.0:
            status = "above"
            label = f"{((ratio-1)*100):.0f}% above avg"
        elif ratio < 0.5:
            status = "low"
            label = f"{((1-ratio)*100):.0f}% below avg"
        else:
            status = "normal"
            label = "Within range"
        results.append({
            "category": cat_name,
            "your_spend": user_spend,
            "avg_india": avg,
            "status": status,
            "label": label,
            "ratio": ratio,
        })
    return sorted(results, key=lambda x: x["ratio"], reverse=True)


def compute_india_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for month, grp in df.groupby("month"):
        income = grp[grp["amount"] > 0]["amount"].sum()
        expenses = grp[grp["amount"] < 0]["amount"].abs().sum()
        balance_end = grp["balance"].iloc[-1] if "balance" in grp.columns else (income - expenses)
        balance_start = grp["balance"].iloc[0] if "balance" in grp.columns else income

        dining = grp[grp["category"] == "Food & Dining"]["abs_amount"].sum()
        entertainment = grp[grp["category"] == "Entertainment"]["abs_amount"].sum()
        shopping = grp[grp["category"] == "Shopping"]["abs_amount"].sum()
        emi = grp[grp["category"] == "EMI & Loans"]["abs_amount"].sum()
        upi_transfers = grp[grp["category"] == "UPI Transfer"]["abs_amount"].sum()

        num_transactions = len(grp)
        avg_daily_spend = expenses / 30
        expense_ratio = expenses / income if income > 0 else 1.0
        discretionary_ratio = (dining + entertainment + shopping) / expenses if expenses > 0 else 0
        emi_ratio = emi / income if income > 0 else 0
        balance_drop = (balance_start - balance_end) / balance_start if balance_start > 0 else 0

        risk = 1 if (expense_ratio > 0.90 or balance_drop > 0.30 or emi_ratio > 0.40) else 0

        records.append({
            "month": month,
            "income": income,
            "expenses": expenses,
            "balance_end": balance_end,
            "dining_spend": dining,
            "entertainment_spend": entertainment,
            "shopping_spend": shopping,
            "emi_spend": emi,
            "upi_transfers": upi_transfers,
            "num_transactions": num_transactions,
            "avg_daily_spend": avg_daily_spend,
            "expense_ratio": expense_ratio,
            "discretionary_ratio": discretionary_ratio,
            "emi_ratio": emi_ratio,
            "balance_drop": balance_drop,
            "risk_label": risk,
        })
    return pd.DataFrame(records)


def generate_india_recommendations(df: pd.DataFrame, risk_score: float, monthly: pd.DataFrame) -> list:
    recs = []
    cat = get_india_category_breakdown(df)
    cat_dict = dict(zip(cat["category"], cat["abs_amount"]))
    latest = monthly.iloc[-1] if not monthly.empty else {}
    savings_rate = latest.get("savings_rate", 0)
    income = latest.get("total_income", 0)

    # Savings rate
    if savings_rate < 10:
        recs.append({"icon": "🚨", "priority": "High",
                     "title": "Critical: Savings Rate Below 10%",
                     "detail": f"You're saving only {savings_rate:.1f}% of income. Aim for 20%+ (₹{income*0.2:,.0f}/month). Start a SIP today — even ₹500/month in ELSS grows significantly over time."})
    elif savings_rate < 20:
        recs.append({"icon": "⚠️", "priority": "Medium",
                     "title": f"Savings Rate at {savings_rate:.1f}% — Below Target",
                     "detail": "The 50/30/20 rule: 50% needs, 30% wants, 20% savings. Small UPI spends add up fast — track them daily."})

    # Dining
    dining = cat_dict.get("Food & Dining", 0)
    bench_dining = INDIA_BENCHMARKS["Food & Dining"]["avg"]
    if dining > bench_dining * 1.5:
        saving = dining - bench_dining
        recs.append({"icon": "🍱", "priority": "Medium",
                     "title": f"Food Delivery Spending ₹{dining:,.0f} — {dining/bench_dining:.1f}x Avg",
                     "detail": f"Average Indian urban household spends ₹{bench_dining:,}/month on dining. You could save ₹{saving:,.0f}/month by cooking 4 meals/week at home. Try Swiggy/Zomato Gold only for special occasions."})

    # EMI
    emi = cat_dict.get("EMI & Loans", 0)
    if income > 0 and emi / income > 0.4:
        recs.append({"icon": "🏦", "priority": "High",
                     "title": f"EMI Load at {emi/income*100:.0f}% of Income — Dangerous",
                     "detail": f"RBI recommends EMIs ≤ 40% of income. You're at {emi/income*100:.0f}%. Consider prepaying high-interest loans (personal loan > car loan > home loan). Avoid new BNPL/credit spending."})

    # Recharge & Bills
    bills = cat_dict.get("Recharge & Bills", 0)
    if bills > 1500:
        recs.append({"icon": "📱", "priority": "Low",
                     "title": "High Mobile/DTH Bills",
                     "detail": f"₹{bills:,.0f}/month on recharges. Consider Jio/BSNL family plans or annual recharges (15-20% cheaper). Cancel unused OTT subscriptions — the average Indian has 3+ active ones."})

    # Investment nudge
    invest = cat_dict.get("Investment", 0)
    if invest == 0 and savings_rate > 10:
        recs.append({"icon": "📈", "priority": "Medium",
                     "title": "No Investments Detected",
                     "detail": "You have savings capacity but no investment transactions found. Start with: ₹500/month ELSS SIP (tax saving u/s 80C), PPF for long-term, and NPS for retirement. Even ₹1,000/month at 12% return = ₹10L in 20 years."})

    # Shopping
    shopping = cat_dict.get("Shopping", 0)
    bench_shop = INDIA_BENCHMARKS["Shopping"]["avg"]
    if shopping > bench_shop * 2:
        recs.append({"icon": "🛍️", "priority": "Medium",
                     "title": f"Shopping ₹{shopping:,.0f} — {shopping/bench_shop:.1f}x Avg Indian Household",
                     "detail": "Try the 72-hour rule before any online purchase. Uninstall shopping apps from your home screen. Use wishlist feature — most items get discounted within weeks."})

    # UPI micro-spends
    upi_count = len(df[(df["amount"] < 0) & (df["abs_amount"] < 200)])
    if upi_count > 30:
        total_micro = df[(df["amount"] < 0) & (df["abs_amount"] < 200)]["abs_amount"].sum()
        recs.append({"icon": "📲", "priority": "Low",
                     "title": f"{upi_count} Small UPI Payments = ₹{total_micro:,.0f} Leakage",
                     "detail": "Small UPI payments under ₹200 are the biggest budget leak. Set a daily UPI limit on your banking app and review these transactions weekly."})

    # Risk
    if risk_score > 0.6:
        recs.append({"icon": "🔴", "priority": "High",
                     "title": "High Risk: May Run Short Before Month End",
                     "detail": "Based on your spending trajectory, you may exhaust funds before salary. Immediate actions: pause all non-essential UPI payments, use cash for groceries to feel the spend, activate spend alerts on your bank app."})

    if not recs:
        recs.append({"icon": "✅", "priority": "Low",
                     "title": "Financial Health Looks Good!",
                     "detail": "You're managing well! Next step: increase SIP amount by 10% every year (step-up SIP), build 6-month emergency fund, and consider term insurance if you haven't already."})

    # Ensure emoji key
    for r in recs:
        r.setdefault("emoji", r["icon"])
    return recs
