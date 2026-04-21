"""
Indian Bank Statement CSV Parser
Supports: HDFC, SBI, ICICI, Axis, Kotak, YES Bank,
          Paytm, PhonePe, GPay export formats
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import io


# ── Indian date formats ───────────────────────────────────────
INDIAN_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y",
    "%d %b %y", "%d-%b-%y",
    "%d %B %Y", "%d-%B-%Y",
    "%Y-%m-%d", "%Y/%m/%d",
    "%d-%b-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
]


def parse_indian_date(val) -> Optional[datetime]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in INDIAN_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True)
    except:
        return None


def clean_amount(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    s = re.sub(r"[₹,\s]", "", s)
    s = s.replace("(", "-").replace(")", "")  # accounting format
    try:
        return float(s)
    except:
        return 0.0


# ── Bank format detectors ─────────────────────────────────────

BANK_SIGNATURES = {
    "HDFC": ["narration", "chq./ref.no.", "withdrawal amt.", "deposit amt.", "closing balance"],
    "SBI": ["txn date", "value date", "description", "ref no./cheque no.", "debit", "credit", "balance"],
    "ICICI": ["transaction date", "transaction remarks", "withdrawal amount (inr )", "deposit amount (inr )", "balance (inr )"],
    "Axis": ["tran date", "particulars", "debit", "credit", "balance"],
    "Kotak": ["description", "debit amount", "credit amount", "balance"],
    "Paytm": ["date", "description", "type", "amount", "closing balance"],
    "PhonePe": ["transaction id", "date", "description", "debit", "credit"],
    "YES": ["date", "transaction details", "cheque number", "debit amount", "credit amount", "balance amount"],
    "PNB": ["date", "particulars", "debit", "credit", "balance"],
    "BOB": ["txn date", "description", "ref number", "debit amount", "credit amount", "balance"],
}

# ── Column name normalizer ────────────────────────────────────

def detect_bank(columns: list) -> str:
    cols_lower = [c.lower().strip() for c in columns]
    col_str = " | ".join(cols_lower)
    scores = {}
    for bank, sigs in BANK_SIGNATURES.items():
        score = sum(1 for s in sigs if s in col_str)
        scores[bank] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Generic"


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Map bank-specific columns to standard: date, description, debit, credit, balance."""
    bank = detect_bank(df.columns.tolist())
    cols = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Mapping rules per bank
    mappings = {
        "HDFC": {
            "date": ["date"], "description": ["narration"],
            "debit": ["withdrawal amt.", "debit"], "credit": ["deposit amt.", "credit"],
            "balance": ["closing balance", "balance"]
        },
        "SBI": {
            "date": ["txn date", "value date", "date"],
            "description": ["description", "particulars"],
            "debit": ["debit", "withdrawal"], "credit": ["credit", "deposit"],
            "balance": ["balance"]
        },
        "ICICI": {
            "date": ["transaction date", "date"],
            "description": ["transaction remarks", "narration", "description"],
            "debit": ["withdrawal amount (inr )", "debit", "withdrawal"],
            "credit": ["deposit amount (inr )", "credit", "deposit"],
            "balance": ["balance (inr )", "balance"]
        },
        "Axis": {
            "date": ["tran date", "date"],
            "description": ["particulars", "description", "narration"],
            "debit": ["debit"], "credit": ["credit"],
            "balance": ["balance"]
        },
        "Kotak": {
            "date": ["date"], "description": ["description", "narration"],
            "debit": ["debit amount", "debit"],
            "credit": ["credit amount", "credit"],
            "balance": ["balance"]
        },
        "Paytm": {
            "date": ["date", "transaction date"],
            "description": ["description", "merchant name", "remarks"],
            "debit": ["debit", "amount paid", "withdrawn"],
            "credit": ["credit", "amount received"],
            "balance": ["closing balance", "wallet balance", "balance"]
        },
        "PhonePe": {
            "date": ["date", "transaction date"],
            "description": ["description", "merchant", "remarks", "comment"],
            "debit": ["debit", "paid", "withdrawal"],
            "credit": ["credit", "received"],
            "balance": ["balance"]
        },
        "Generic": {
            "date": ["date", "txn date", "transaction date", "value date"],
            "description": ["description", "narration", "particulars", "remarks",
                            "transaction details", "merchant"],
            "debit": ["debit", "withdrawal", "dr", "amount (dr)", "debit amount",
                      "withdrawal amt.", "paid"],
            "credit": ["credit", "deposit", "cr", "amount (cr)", "credit amount",
                       "deposit amt.", "received"],
            "balance": ["balance", "closing balance", "available balance"]
        }
    }

    m = mappings.get(bank, mappings["Generic"])
    rename_map = {}
    current_cols = list(df.columns)

    for std_name, candidates in m.items():
        for candidate in candidates:
            if candidate in current_cols and std_name not in rename_map.values():
                rename_map[candidate] = std_name
                break

    df.rename(columns=rename_map, inplace=True)
    return df, bank


# ── UPI/NEFT description categorizer ─────────────────────────

DESCRIPTION_RULES = {
    "Food & Dining": [
        "zomato", "swiggy", "dominos", "pizza", "kfc", "mcdonalds", "burger",
        "restaurant", "cafe", "dhaba", "hotel", "food", "eat", "dining",
        "dunzo", "magicpin", "dineout"
    ],
    "Groceries": [
        "bigbasket", "blinkit", "grofers", "jiomart", "dmart", "more supermarket",
        "reliance fresh", "star bazaar", "easyday", "grocery", "kirana", "vegetable",
        "fruits", "milk", "dairy", "nature's basket"
    ],
    "Transport": [
        "ola", "uber", "rapido", "auto", "cab", "taxi", "metro", "bus",
        "yulu", "bounce", "vroom", "petrol", "fuel", "hp ", "ioc", "bpcl",
        "irctc", "railway", "train", "redbus", "abhibus"
    ],
    "Travel": [
        "makemytrip", "goibibo", "yatra", "cleartrip", "airasia", "indigo",
        "spicejet", "air india", "vistara", "hotel booking", "oyo", "treebo"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "shopsy",
        "snapdeal", "jabong", "pepperfry", "urban ladder", "ikea", "decathlon",
        "tatacliq", "croma", "vijay sales", "reliance digital"
    ],
    "Recharge & Bills": [
        "airtel", "jio", "bsnl", "vodafone", "vi ", "recharge", "dth",
        "tataplay", "dishtv", "sun direct", "dish", "mobile bill", "phone bill",
        "prepaid", "postpaid", "broadband"
    ],
    "Utilities": [
        "electricity", "bescom", "msedcl", "tata power", "adani electric",
        "water bill", "gas", "piped gas", "igl", "mgl", "mahanagar gas",
        "bwssb", "delhi jal", "utility", "maintenance", "society"
    ],
    "Health": [
        "pharmeasy", "netmeds", "1mg", "apollo pharmacy", "medplus",
        "hospital", "clinic", "doctor", "medical", "lab", "diagnostic",
        "chemist", "pharmacy", "medicine", "health", "practo"
    ],
    "Entertainment": [
        "netflix", "amazon prime", "hotstar", "disney", "spotify", "youtube",
        "bookmyshow", "pvr", "inox", "movie", "game", "gaming", "steam",
        "sonyliv", "zee5", "voot", "jiosaavn", "gaana"
    ],
    "Education": [
        "byju", "unacademy", "vedantu", "coursera", "udemy", "upgrad",
        "school fee", "college fee", "tuition", "coaching", "exam fee",
        "board", "university", "institute"
    ],
    "Investment": [
        "groww", "zerodha", "upstox", "angel", "5paisa", "icicidirect",
        "hdfcsec", "sbimf", "mutual fund", "sip", "ppf", "nps", "elss",
        "stocks", "shares", "demat"
    ],
    "Insurance": [
        "lic", "lici", "hdfc life", "icici pru", "bajaj allianz",
        "max life", "term insurance", "health insurance", "car insurance",
        "vehicle insurance", "premium"
    ],
    "EMI & Loans": [
        "emi", "loan", "credit card", "bajaj finance", "home loan",
        "personal loan", "car loan", "two wheeler", "bnpl", "lazypay",
        "simpl", "zestmoney"
    ],
    "Savings & Transfer": [
        "transfer", "neft", "rtgs", "imps", "savings", "fd ", "fixed deposit",
        "rd ", "recurring", "sweep"
    ],
}


def categorize_indian(description: str, vpa: str = "") -> str:
    text = f"{description} {vpa}".lower()
    for category, keywords in DESCRIPTION_RULES.items():
        if any(k in text for k in keywords):
            return category

    # UPI peer-to-peer patterns
    if re.search(r"upi[-/]?\d+|@\w+|p2p|peer", text):
        return "UPI Transfer"
    if re.search(r"salary|stipend|payroll|wages", text):
        return "Income"
    if re.search(r"rent|pg |paying guest|landlord|hostel", text):
        return "Housing & Rent"
    if re.search(r"atm|cash withdrawal|cash deposit", text):
        return "Cash"

    return "Others"


# ── Main parsing function ─────────────────────────────────────

def parse_bank_csv(file_obj) -> Tuple[pd.DataFrame, str, str]:
    """
    Parse an Indian bank statement CSV.
    Returns: (dataframe, bank_name, error_message)
    """
    try:
        # Try reading with different encodings / skip rows
        content = None
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                if hasattr(file_obj, "read"):
                    file_obj.seek(0)
                    raw = file_obj.read()
                    if isinstance(raw, bytes):
                        content = raw.decode(enc, errors="replace")
                    else:
                        content = raw
                    break
                else:
                    with open(file_obj, "r", encoding=enc, errors="replace") as f:
                        content = f.read()
                    break
            except:
                continue

        if content is None:
            return pd.DataFrame(), "Unknown", "Could not read file"

        # Find the header row (skip bank metadata rows at top)
        lines = content.splitlines()
        header_row = 0
        date_keywords = ["date", "txn", "transaction", "value date"]
        for i, line in enumerate(lines[:20]):
            if any(k in line.lower() for k in date_keywords):
                header_row = i
                break

        df_raw = pd.read_csv(
            io.StringIO("\n".join(lines[header_row:])),
            skip_blank_lines=True,
            on_bad_lines="skip",
        )
        df_raw.dropna(how="all", inplace=True)
        df_raw.dropna(axis=1, how="all", inplace=True)

        df, bank = normalize_columns(df_raw)

        # Validate essential columns exist
        if "date" not in df.columns:
            return pd.DataFrame(), bank, "Could not find date column. Please check CSV format."

        # Parse dates
        df["date"] = df["date"].apply(parse_indian_date)
        df.dropna(subset=["date"], inplace=True)

        # Build amount column from debit/credit
        if "amount" not in df.columns:
            df["debit"] = df["debit"].apply(clean_amount) if "debit" in df.columns else 0
            df["credit"] = df["credit"].apply(clean_amount) if "credit" in df.columns else 0
            df["amount"] = df["credit"] - df["debit"]
        else:
            df["amount"] = df["amount"].apply(clean_amount)

        # Description
        if "description" not in df.columns:
            for c in df.columns:
                if c not in ["date", "amount", "debit", "credit", "balance"]:
                    df["description"] = df[c].astype(str)
                    break
        df["description"] = df.get("description", pd.Series(["Transaction"] * len(df))).fillna("Transaction").astype(str)

        # Balance
        if "balance" in df.columns:
            df["balance"] = df["balance"].apply(clean_amount)
        else:
            df["balance"] = df["amount"].cumsum() + 10000

        # Category
        df["category"] = df["description"].apply(lambda d: categorize_indian(d))

        # Fix income rows
        df.loc[df["amount"] > 0, "category"] = df.loc[df["amount"] > 0, "description"].apply(
            lambda d: "Income" if any(k in d.lower() for k in ["salary", "credit", "income", "interest", "refund"])
            else categorize_indian(d)
        )

        df["type"] = df["amount"].apply(lambda x: "credit" if x >= 0 else "debit")
        df["abs_amount"] = df["amount"].abs()
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["day_of_month"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.day_name()
        df["source"] = "Bank CSV"
        df["bank"] = bank

        # Filter out zero-amount rows
        df = df[df["abs_amount"] > 0].copy()
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df, bank, ""

    except Exception as e:
        return pd.DataFrame(), "Unknown", str(e)


def generate_sample_sms() -> str:
    """Return sample SMS messages for demo/testing."""
    return """Dear Customer, INR 450.00 debited from A/c XX5678 on 21-04-2024 to VPA zomato@icici UPI Ref 412345678901. Available Balance: INR 24,550.00

Your A/c XX5678 is credited with INR 35,000.00 on 01-04-2024 by NEFT from EMPLOYER INDIA LTD. UPI Ref:112345. Avl Bal:INR 55,000.00

Rs.320.00 debited from A/c XX5678 on 15-04-2024 to VPA swiggy@axisbank UPI Ref 312345. Avl Bal:INR 48,200.00

INR 1,200.00 debited from A/c XX5678 on 01-04-2024 to VPA landlord123@oksbi. Rent Payment. UPI Ref 212345.

Dear Customer Rs.199.00 debited from your A/c XX5678 on 05-04-2024 to VPA netflix@icici. Avl Bal: 52,301.00

INR 85.00 debited from A/c XX5678 on 08-04-2024 to VPA rapido@kotak UPI Ref 512345. Avl Bal:INR 47,600.00

Your A/c XX5678 credited with INR 2,500.00 on 10-04-2024. UPI transfer from RAHUL SHARMA. Ref 612345.

Rs.2,500.00 debited from A/c XX5678 on 02-04-2024 to VPA hdfcbank@hdfc. EMI payment. UPI Ref 712345.

INR 650.00 debited from A/c XX5678 on 12-04-2024 to VPA bigbasket@hdfcbank UPI Ref 812345. Avl Bal:INR 44,850.00

Dear Customer, Rs.120.00 debited from A/c XX5678 on 14-04-2024 to VPA uber@razorpay UPI Ref 912345. Avl Bal:INR 44,730.00

INR 499.00 debited from A/c XX5678 on 16-04-2024 to VPA hotstar@axisbank UPI Ref 102345. Avl Bal:INR 44,231.00

Rs.350.00 debited from A/c XX5678 on 18-04-2024 to VPA pharmeasy@icici UPI Ref 112346. Avl Bal:INR 43,881.00

INR 5,000.00 debited from A/c XX5678 on 20-04-2024 to VPA savings@sbi. Transfer to savings. UPI Ref 122346.

Rs.180.00 debited from A/c XX5678 on 22-04-2024 to VPA dominos@hdfcbank UPI Ref 132346. Avl Bal:INR 38,701.00

INR 799.00 debited from A/c XX5678 on 24-04-2024 to VPA amazon@apl UPI Ref 142346. Shopping. Avl Bal:INR 37,902.00"""
