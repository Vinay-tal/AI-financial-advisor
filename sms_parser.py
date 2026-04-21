"""
Indian Bank SMS Transaction Parser
Supports: HDFC, SBI, ICICI, Axis, Kotak, YES Bank,
          PhonePe, GPay, Paytm, IDFC, PNB, BOB, Canara
"""

import re
import pandas as pd
from datetime import datetime
from typing import Optional


# ── UPI VPA → Merchant category map ──────────────────────────
VPA_CATEGORY_MAP = {
    # Food & Dining
    "zomato": "Food & Dining", "swiggy": "Food & Dining",
    "dominos": "Food & Dining", "mcdonalds": "Food & Dining",
    "kfc": "Food & Dining", "pizzahut": "Food & Dining",
    "dunzo": "Food & Dining", "blinkit": "Groceries",
    "bigbasket": "Groceries", "grofers": "Groceries",
    "jiomart": "Groceries", "dmart": "Groceries",
    # Shopping
    "amazon": "Shopping", "flipkart": "Shopping",
    "myntra": "Shopping", "ajio": "Shopping",
    "meesho": "Shopping", "snapdeal": "Shopping",
    "nykaa": "Shopping", "tatacliq": "Shopping",
    # Transport
    "ola": "Transport", "uber": "Transport",
    "rapido": "Transport", "yulu": "Transport",
    "irctc": "Transport", "redbus": "Transport",
    "makemytrip": "Travel", "goibibo": "Travel",
    "cleartrip": "Travel", "airasia": "Travel",
    "indigo": "Travel", "spicejet": "Travel",
    # Recharge & Bills
    "airtel": "Recharge & Bills", "jio": "Recharge & Bills",
    "bsnl": "Recharge & Bills", "vodafone": "Recharge & Bills",
    "tataplay": "Recharge & Bills", "dishtv": "Recharge & Bills",
    "bescom": "Utilities", "msedcl": "Utilities",
    "mahadiscom": "Utilities", "torrent": "Utilities",
    # Health
    "pharmeasy": "Health", "netmeds": "Health",
    "1mg": "Health", "apollo": "Health",
    "practo": "Health", "lybrate": "Health",
    # Entertainment
    "netflix": "Entertainment", "hotstar": "Entertainment",
    "spotify": "Entertainment", "sonyliv": "Entertainment",
    "zeenow": "Entertainment", "primevideo": "Entertainment",
    "bookmyshow": "Entertainment",
    # Finance
    "groww": "Investment", "zerodha": "Investment",
    "upstox": "Investment", "paytmmoney": "Investment",
    "lici": "Insurance", "hdfclife": "Insurance",
    "icicipru": "Insurance", "sbimf": "Investment",
    "hdfc": "Banking", "icici": "Banking",
    "sbi": "Banking", "axis": "Banking",
    "kotak": "Banking", "yesbank": "Banking",
    # Education
    "byju": "Education", "unacademy": "Education",
    "vedantu": "Education", "coursera": "Education",
    "udemy": "Education",
}

# ── Bank-specific SMS regex patterns ─────────────────────────
SMS_PATTERNS = [
    # HDFC Bank
    {
        "bank": "HDFC",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*)\s+(?:debited|credited|sent|received).*?(?:A/c|Acct?\.?)\s*[Xx*]+(\d{4}).*?(?:on|dated?)?\s*(\d{1,2}[-/]\w{2,3}[-/]\d{2,4})",
        "amount_group": 1, "account_group": 2, "date_group": 3, "type_word": "debited"
    },
    # HDFC UPI
    {
        "bank": "HDFC",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*)\s+(?:debited|credited).*?VPA\s+([\w.\-@]+).*?(\d{2}[-/]\d{2}[-/]\d{2,4})",
        "amount_group": 1, "vpa_group": 2, "date_group": 3, "type_word": "debited"
    },
    # SBI
    {
        "bank": "SBI",
        "pattern": r"(?:Rs\.?|INR|Amt)\s*:?\s*([\d,]+\.?\d*).*?(?:debited|credited|withdrawn|deposited).*?(?:A/[Cc]|Acct?)\.?\s*[Xx*]+(\w{4,6}).*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        "amount_group": 1, "account_group": 2, "date_group": 3, "type_word": "debited"
    },
    # ICICI
    {
        "bank": "ICICI",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*)\s+(?:debited|credited).*?(?:A/c|account)\s*[Xx*]+(\d{4}).*?(\d{2}-\d{2}-\d{4})",
        "amount_group": 1, "account_group": 2, "date_group": 3, "type_word": "debited"
    },
    # ICICI UPI
    {
        "bank": "ICICI",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*).*?(?:sent to|received from)\s+([\w.\-@]+).*?(?:UPI|Ref).*?(\d{2}[-/]\d{2}[-/]\d{2,4})",
        "amount_group": 1, "vpa_group": 2, "date_group": 3, "type_word": "sent"
    },
    # Axis Bank
    {
        "bank": "Axis",
        "pattern": r"(?:Rs\.?|INR|Amount)\s*:?\s*([\d,]+\.?\d*).*?(?:debited|credited).*?(?:A/c|Acct?)\.?\s*[Xx*]+(\d{4}).*?(\d{1,2}[-/]\w{3}[-/]\d{2,4})",
        "amount_group": 1, "account_group": 2, "date_group": 3, "type_word": "debited"
    },
    # Kotak
    {
        "bank": "Kotak",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*).*?(?:debited|credited).*?(\d{4}).*?(\d{2}[-/]\d{2}[-/]\d{4})",
        "amount_group": 1, "account_group": 2, "date_group": 3, "type_word": "debited"
    },
    # PhonePe
    {
        "bank": "PhonePe",
        "pattern": r"(?:Rs\.?|INR)\s*([\d,]+\.?\d*)\s+(?:paid to|sent to|received from)\s+(.+?)\s+(?:on|via|using).*?(\d{1,2}[-/\s]\w{2,9}[-/\s]\d{2,4})",
        "amount_group": 1, "merchant_group": 2, "date_group": 3, "type_word": "paid"
    },
    # Generic UPI
    {
        "bank": "UPI",
        "pattern": r"(?:Rs\.?|INR|₹)\s*([\d,]+\.?\d*).*?(?:debited|paid|sent).*?(?:UPI|upi).*?(?:VPA|vpa|to)\s*([\w.\-@]+)",
        "amount_group": 1, "vpa_group": 2, "date_group": None, "type_word": "debited"
    },
    # Generic debit/credit
    {
        "bank": "Generic",
        "pattern": r"(?:Rs\.?|INR|₹)\s*([\d,]+\.?\d*)\s+(debited|credited).*?(?:on\s+)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        "amount_group": 1, "type_group": 2, "date_group": 3, "type_word": None
    },
]


def parse_amount(text: str) -> float:
    """Clean and convert amount string to float."""
    cleaned = re.sub(r"[,\s₹]", "", text)
    try:
        return float(cleaned)
    except:
        return 0.0


def parse_indian_date(date_str: str) -> Optional[datetime]:
    """Parse various Indian date formats."""
    if not date_str:
        return None
    date_str = date_str.strip()
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
        "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y",
        "%d %B %Y", "%d-%B-%Y",
        "%d %b %y", "%d-%b-%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None


def extract_vpa_category(vpa: str) -> str:
    """Map UPI VPA to spending category."""
    if not vpa:
        return "UPI Transfer"
    vpa_lower = vpa.lower()
    for keyword, category in VPA_CATEGORY_MAP.items():
        if keyword in vpa_lower:
            return category
    # Heuristics
    if "@" in vpa:
        parts = vpa.split("@")
        handle = parts[0].lower()
        for keyword, category in VPA_CATEGORY_MAP.items():
            if keyword in handle:
                return category
    return "UPI Transfer"


def parse_single_sms(sms_text: str) -> Optional[dict]:
    """Parse a single SMS and return transaction dict or None."""
    sms_lower = sms_text.lower()

    # Skip OTPs, alerts, non-transaction SMS
    skip_keywords = ["otp", "password", "login", "verify", "code is", "do not share",
                     "offer", "congratulations", "reward", "cashback alert only"]
    if any(k in sms_lower for k in skip_keywords):
        return None

    # Must contain financial keywords
    financial_keywords = ["debited", "credited", "paid", "received", "transferred",
                          "withdrawn", "deposited", "rs.", "inr", "₹", "upi", "neft", "imps"]
    if not any(k in sms_lower for k in financial_keywords):
        return None

    for pat in SMS_PATTERNS:
        match = re.search(pat["pattern"], sms_text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        # Extract amount
        amt_raw = match.group(pat["amount_group"])
        amount = parse_amount(amt_raw)
        if amount <= 0:
            continue

        # Determine type (debit/credit)
        type_word = pat.get("type_word")
        if type_word is None and "type_group" in pat:
            try:
                type_word = match.group(pat["type_group"]).lower()
            except:
                type_word = "debited"

        is_credit = type_word in ["credited", "received", "deposited"] if type_word else False
        if "credit" in sms_lower and "debit" not in sms_lower:
            is_credit = True

        # Extract date
        date = datetime.today()
        if pat.get("date_group"):
            try:
                date_str = match.group(pat["date_group"])
                parsed = parse_indian_date(date_str)
                if parsed:
                    date = parsed
            except:
                pass

        # Extract VPA / merchant
        description = "UPI Transaction"
        category = "UPI Transfer"

        if "vpa_group" in pat:
            try:
                vpa = match.group(pat["vpa_group"])
                description = f"UPI - {vpa}"
                category = extract_vpa_category(vpa)
            except:
                pass
        elif "merchant_group" in pat:
            try:
                merchant = match.group(pat["merchant_group"]).strip()
                description = merchant
                category = extract_vpa_category(merchant)
            except:
                pass

        # Extract account last 4
        account = ""
        if "account_group" in pat:
            try:
                account = f"XX{match.group(pat['account_group'])}"
            except:
                pass

        signed_amount = amount if is_credit else -amount

        return {
            "date": date,
            "description": description,
            "amount": signed_amount,
            "abs_amount": amount,
            "type": "credit" if is_credit else "debit",
            "category": "Income" if is_credit else category,
            "bank": pat["bank"],
            "account": account,
            "source": "SMS",
        }

    return None


def parse_sms_file(content: str) -> pd.DataFrame:
    """
    Parse a plain text file where each line (or block) is one SMS.
    Supports both line-by-line and blank-line-separated formats.
    """
    # Try blank-line separated first
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    if len(blocks) < 3:
        # Fall back to line-by-line
        blocks = [l.strip() for l in content.splitlines() if l.strip()]

    records = []
    for block in blocks:
        result = parse_single_sms(block)
        if result:
            records.append(result)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add derived columns
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_month"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()

    # Compute running balance (estimate)
    df["balance"] = df["amount"].cumsum() + 10000  # assume 10k starting balance

    return df


def parse_sms_dataframe(df_sms: pd.DataFrame) -> pd.DataFrame:
    """Parse SMS from a DataFrame with a 'message' or 'body' column."""
    col = None
    for c in ["message", "body", "text", "sms", "Body", "Message"]:
        if c in df_sms.columns:
            col = c
            break
    if col is None:
        return pd.DataFrame()

    records = []
    for _, row in df_sms.iterrows():
        result = parse_single_sms(str(row[col]))
        if result:
            # Try to get date from df columns
            for dc in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
                if dc in df_sms.columns:
                    try:
                        result["date"] = pd.to_datetime(row[dc])
                    except:
                        pass
                    break
            records.append(result)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_month"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df["balance"] = df["amount"].cumsum() + 10000
    return df
