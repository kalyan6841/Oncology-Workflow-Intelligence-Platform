from __future__ import annotations

import random
from typing import Dict, Tuple

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _heuristic_scores(row: pd.Series) -> Tuple[float, float]:
    delay_score = 0.12
    if str(row.get("lab_status", "")).lower() != "ready":
        delay_score += 0.42
    if str(row.get("pap_status", "")).lower() != "ready":
        delay_score += 0.28
    if _safe_float(row.get("chair_hours"), 3.0) >= 5:
        delay_score += 0.08
    if _safe_int(row.get("cycle_number"), 1) >= 5:
        delay_score += 0.06
    if str(row.get("treatment_intent", "")).lower() == "palliative":
        delay_score += 0.05

    no_show_score = 0.10
    if _safe_int(row.get("age"), 50) >= 65:
        no_show_score += 0.10
    if str(row.get("ward_assigned", "")).strip() == "":
        no_show_score += 0.10
    if _safe_float(row.get("chair_hours"), 3.0) > 4:
        no_show_score += 0.12
    if str(row.get("lab_status", "")).lower() != "ready":
        no_show_score += 0.07

    return min(max(delay_score, 0.02), 0.98), min(max(no_show_score, 0.02), 0.98)


def _build_synthetic_training(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    rows = []
    sample = df.copy()
    if sample.empty:
        return pd.DataFrame()
    for _ in range(n):
        base = sample.sample(1, replace=True).iloc[0].to_dict()
        base["age"] = max(18, _safe_int(base.get("age"), 50) + random.randint(-6, 6))
        base["cycle_number"] = max(1, _safe_int(base.get("cycle_number"), 2) + random.randint(-1, 2))
        base["chair_hours"] = max(0.0, _safe_float(base.get("chair_hours"), 3.0) + random.uniform(-0.8, 1.1))
        delay_prob, no_show_prob = _heuristic_scores(pd.Series(base))
        base["delay_label"] = 1 if random.random() < delay_prob else 0
        base["no_show_label"] = 1 if random.random() < no_show_prob else 0
        rows.append(base)
    return pd.DataFrame(rows)


def _train_models(train_df: pd.DataFrame):
    numeric = ["age", "cycle_number", "chair_hours", "frequency_days"]
    categorical = ["cancer_type", "stage", "biomarker", "treatment_intent", "lab_status", "pap_status"]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    delay_model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    no_show_model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    X = train_df[numeric + categorical]
    delay_model.fit(X, train_df["delay_label"])
    no_show_model.fit(X, train_df["no_show_label"])
    return delay_model, no_show_model


def compute_ml_signals(patients_df: pd.DataFrame, timeline_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    merged = patients_df.merge(
        timeline_df[["patient_id", "readiness", "safe_window_date", "next_cycle_date", "chair_hours"]],
        on="patient_id",
        how="left",
        suffixes=("_patient", "_timeline"),
    )
    if merged.empty:
        return merged, {"expected_next_day_patients": 0.0, "expected_next_day_chair_hours": 0.0}

    if "chair_hours_timeline" in merged.columns:
        merged["chair_hours"] = merged["chair_hours_timeline"]
    elif "chair_hours_patient" in merged.columns:
        merged["chair_hours"] = merged["chair_hours_patient"]
    elif "chair_hours" not in merged.columns:
        merged["chair_hours"] = 3.0
    merged["chair_hours"] = merged["chair_hours"].fillna(3.0)
    merged["frequency_days"] = merged["frequency_days"].fillna(21)
    merged["age"] = merged["age"].fillna(50)
    merged["cycle_number"] = merged["cycle_number"].fillna(1)

    if SKLEARN_AVAILABLE and len(merged) >= 5:
        train = _build_synthetic_training(merged, n=300)
        delay_model, no_show_model = _train_models(train)
        feat_cols = ["age", "cycle_number", "chair_hours", "frequency_days", "cancer_type", "stage", "biomarker", "treatment_intent", "lab_status", "pap_status"]
        merged["cycle_delay_risk"] = delay_model.predict_proba(merged[feat_cols])[:, 1]
        merged["no_show_risk"] = no_show_model.predict_proba(merged[feat_cols])[:, 1]
    else:
        scores = merged.apply(_heuristic_scores, axis=1)
        merged["cycle_delay_risk"] = scores.apply(lambda x: x[0])
        merged["no_show_risk"] = scores.apply(lambda x: x[1])

    merged["cycle_delay_risk_pct"] = (merged["cycle_delay_risk"] * 100).round(1)
    merged["no_show_risk_pct"] = (merged["no_show_risk"] * 100).round(1)
    merged["workflow_priority"] = merged["cycle_delay_risk"].apply(
        lambda v: "High" if v >= 0.65 else ("Medium" if v >= 0.35 else "Low")
    )
    merged["workflow_recommendation"] = merged.apply(
        lambda r: (
            "Immediate pre-cycle follow-up call + lab escalation"
            if r["cycle_delay_risk"] >= 0.65
            else ("Reminder + readiness review in 24h" if r["cycle_delay_risk"] >= 0.35 else "Standard workflow")
        ),
        axis=1,
    )

    expected_next_day_patients = float((1 - merged["cycle_delay_risk"]).sum())
    expected_next_day_chair_hours = float(((1 - merged["cycle_delay_risk"]) * merged["chair_hours"]).sum())
    forecast = {
        "expected_next_day_patients": round(expected_next_day_patients, 1),
        "expected_next_day_chair_hours": round(expected_next_day_chair_hours, 1),
    }
    return merged, forecast

