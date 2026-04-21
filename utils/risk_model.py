"""
Risk Prediction Model — predicts if user will run out of money before month end.
Uses Random Forest with engineered features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
import os
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.joblib")
FEATURE_COLS = [
    "expense_ratio",
    "discretionary_ratio",
    "avg_daily_spend",
    "balance_drop",
    "dining_spend",
    "shopping_spend",
    "entertainment_spend",
    "num_transactions",
]


@st.cache_data(show_spinner=False)
def train_model(features_df: pd.DataFrame):
    """Train risk prediction model on monthly features."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X = features_df[FEATURE_COLS].fillna(0)
    y = features_df["risk_label"]

    # Need at least 2 samples with both classes; if not, use synthetic augmentation
    if len(X) < 4 or y.nunique() < 2:
        X, y = _augment_data(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def _augment_data(X: pd.DataFrame, y: pd.Series):
    """Synthetically augment small datasets for training."""
    np.random.seed(42)
    n_synthetic = max(20, len(X) * 4)
    
    # Generate synthetic risky months
    risky = pd.DataFrame({
        "expense_ratio": np.random.uniform(0.85, 1.2, n_synthetic // 2),
        "discretionary_ratio": np.random.uniform(0.4, 0.7, n_synthetic // 2),
        "avg_daily_spend": np.random.uniform(80, 180, n_synthetic // 2),
        "balance_drop": np.random.uniform(0.25, 0.6, n_synthetic // 2),
        "dining_spend": np.random.uniform(300, 700, n_synthetic // 2),
        "shopping_spend": np.random.uniform(400, 900, n_synthetic // 2),
        "entertainment_spend": np.random.uniform(200, 500, n_synthetic // 2),
        "num_transactions": np.random.randint(25, 60, n_synthetic // 2),
    })
    risky_labels = pd.Series([1] * (n_synthetic // 2))

    # Generate synthetic safe months
    safe = pd.DataFrame({
        "expense_ratio": np.random.uniform(0.4, 0.75, n_synthetic // 2),
        "discretionary_ratio": np.random.uniform(0.1, 0.3, n_synthetic // 2),
        "avg_daily_spend": np.random.uniform(20, 70, n_synthetic // 2),
        "balance_drop": np.random.uniform(-0.1, 0.15, n_synthetic // 2),
        "dining_spend": np.random.uniform(50, 200, n_synthetic // 2),
        "shopping_spend": np.random.uniform(50, 200, n_synthetic // 2),
        "entertainment_spend": np.random.uniform(20, 100, n_synthetic // 2),
        "num_transactions": np.random.randint(10, 30, n_synthetic // 2),
    })
    safe_labels = pd.Series([0] * (n_synthetic // 2))

    X_aug = pd.concat([X, risky, safe], ignore_index=True)
    y_aug = pd.concat([y, risky_labels, safe_labels], ignore_index=True)
    return X_aug, y_aug


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_risk(model, current_features: dict) -> dict:
    """Predict risk for current period."""
    feat_df = pd.DataFrame([current_features])[FEATURE_COLS].fillna(0)
    prob = model.predict_proba(feat_df)[0]
    risk_prob = prob[1] if len(prob) > 1 else prob[0]
    label = model.predict(feat_df)[0]

    if risk_prob < 0.35:
        level = "Low"
        color = "green"
        emoji = "🟢"
    elif risk_prob < 0.65:
        level = "Medium"
        color = "orange"
        emoji = "🟡"
    else:
        level = "High"
        color = "red"
        emoji = "🔴"

    return {
        "risk_probability": float(risk_prob),
        "risk_label": int(label),
        "risk_level": level,
        "color": color,
        "emoji": emoji,
    }


def get_feature_importance(model) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    return pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)
