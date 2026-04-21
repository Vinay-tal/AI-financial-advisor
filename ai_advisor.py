"""
AI Financial Advisor — powered by Groq LLM (llama3-70b-8192)
Falls back to rule-based advice if no API key is set.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def build_financial_context(df, monthly_summary, risk_result, category_breakdown) -> str:
    """Build a rich context string from user financial data."""
    latest = monthly_summary.iloc[-1] if not monthly_summary.empty else {}
    top_cats = category_breakdown.head(5).to_dict("records")
    cat_str = "\n".join([f"  - {r['category']}: ${r['abs_amount']:.2f} ({r['percentage']:.1f}%)" for r in top_cats])

    context = f"""
USER FINANCIAL PROFILE:
- Total transactions analyzed: {len(df)}
- Date range: {df['date'].min().strftime('%b %d, %Y')} to {df['date'].max().strftime('%b %d, %Y')}
- Latest month income: ${latest.get('total_income', 0):.2f}
- Latest month expenses: ${latest.get('total_expenses', 0):.2f}
- Net cash flow: ${latest.get('net_flow', 0):.2f}
- Savings rate: {latest.get('savings_rate', 0):.1f}%
- Financial risk level: {risk_result.get('risk_level', 'Unknown')} ({risk_result.get('risk_probability', 0)*100:.0f}% probability)

TOP SPENDING CATEGORIES:
{cat_str}
"""
    return context.strip()


def get_ai_response(user_message: str, financial_context: str, chat_history: list) -> str:
    """Get response from Groq LLM or fallback to rule-based."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return _fallback_response(user_message, financial_context)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        system_prompt = f"""You are FinSight AI, an expert personal financial advisor with deep expertise in budgeting, saving, investing, and risk management.

You have access to the user's real financial data:

{financial_context}

Guidelines:
- Be concise, direct, and actionable (max 200 words per response)
- Use specific numbers from their data when relevant
- Prioritize the most impactful advice
- Be empathetic but honest about concerning patterns
- Format with bullet points when listing multiple items
- Never give generic advice — always tie it to their specific data
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        return _fallback_response(user_message, financial_context) + f"\n\n*[Note: AI advisor offline — {str(e)[:60]}]*"


def _fallback_response(user_message: str, context: str) -> str:
    """Rule-based fallback when no Groq API key."""
    msg = user_message.lower()

    if any(w in msg for w in ["save", "saving", "savings"]):
        return (
            "💡 **Savings Strategy:**\n\n"
            "• **50/30/20 Rule**: Allocate 50% to needs, 30% to wants, 20% to savings\n"
            "• Automate savings on payday before you spend\n"
            "• Even saving $5/day = $1,825/year\n"
            "• Build a 3-6 month emergency fund first\n\n"
            "Based on your data, your current savings rate and top expenses are shown in the dashboard above."
        )
    elif any(w in msg for w in ["risk", "budget", "month", "run out"]):
        return (
            "⚠️ **Budget Risk Assessment:**\n\n"
            "Your risk score is calculated based on:\n"
            "• Expense-to-income ratio\n"
            "• Discretionary spending levels\n"
            "• Balance trend over time\n\n"
            "To reduce risk: cut dining out by 30%, pause non-essential shopping, and track daily spending."
        )
    elif any(w in msg for w in ["invest", "stock", "fund", "wealth"]):
        return (
            "📈 **Investment Basics:**\n\n"
            "• First: clear high-interest debt (credit cards)\n"
            "• Build 3-month emergency fund\n"
            "• Then consider: Index funds (low cost, diversified)\n"
            "• SIPs in mutual funds starting at ₹500/month\n"
            "• Time in market > timing the market\n\n"
            "*Note: This is educational, not financial advice. Consult a SEBI-registered advisor.*"
        )
    elif any(w in msg for w in ["food", "dining", "eat", "restaurant"]):
        return (
            "🍽️ **Reduce Dining Costs:**\n\n"
            "• Meal prep Sunday: saves 40-60% vs restaurants\n"
            "• Set a monthly dining budget and track it\n"
            "• Use apps for deals (Zomato, Swiggy discounts)\n"
            "• Coffee at home: saves ~$60-100/month\n"
            "• Cook once, eat twice strategy"
        )
    else:
        return (
            "👋 **FinSight AI — Financial Advisor**\n\n"
            "I can help you with:\n"
            "• 💰 Savings strategies\n"
            "• 📊 Budget optimization\n"
            "• ⚠️ Risk reduction\n"
            "• 🍽️ Spending category advice\n"
            "• 📈 Investment basics\n\n"
            "Ask me anything about your finances! For personalized AI advice, add your **Groq API key** in the `.env` file.\n\n"
            "*Get a free key at console.groq.com*"
        )
