from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd


INFUSION_DURATION_MAP: Dict[str, float] = {
    "AC": 2.0,
    "Paclitaxel": 3.0,
    "FOLFOX": 4.0,
    "R-CHOP": 5.0,
    "TCH": 6.0,
    "ABVD": 3.0,
    "VRd": 4.0,
    "Osimertinib": 0.0,  # outpatient oral, no chair block
}


def regimen_duration_hours(regimen_text: str, fallback: float = 3.0) -> float:
    text = str(regimen_text).lower()
    for regimen, duration in INFUSION_DURATION_MAP.items():
        if regimen.lower() in text:
            return duration
    return fallback


def classify_freeing_bucket(minutes_remaining: float) -> str:
    if minutes_remaining <= 30:
        return "Free in 30 minutes"
    if minutes_remaining <= 60:
        return "Free in 60 minutes"
    return "Free in 2 hours"


def build_chair_allocation(
    timeline_df: pd.DataFrame,
    total_chairs: int,
    now: datetime | None = None,
) -> Tuple[pd.DataFrame, dict]:
    current = now or datetime.now()
    timeline = timeline_df.copy()
    timeline["chair_hours"] = timeline["chair_hours"].astype(float)
    timeline["safe_window_date"] = pd.to_datetime(timeline["safe_window_date"])
    timeline["slot_start"] = timeline["safe_window_date"].dt.normalize() + pd.Timedelta(hours=9)
    timeline["slot_end"] = timeline["slot_start"] + pd.to_timedelta(timeline["chair_hours"], unit="h")

    timeline = timeline.sort_values(["slot_start", "patient_id"]).reset_index(drop=True)
    timeline["chair_id"] = (timeline.index % max(total_chairs, 1)) + 1
    timeline["reservation_status"] = "Reserved"

    def chair_state(row: pd.Series) -> str:
        if row["slot_start"] <= current <= row["slot_end"]:
            return "Occupied"
        if current < row["slot_start"]:
            return "Reserved"
        return "Completed"

    timeline["chair_state"] = timeline.apply(chair_state, axis=1)
    timeline["minutes_remaining"] = (timeline["slot_end"] - current).dt.total_seconds() / 60.0
    timeline["free_prediction"] = timeline["minutes_remaining"].apply(
        lambda mins: classify_freeing_bucket(max(mins, 0)) if mins >= 0 else "Free now"
    )

    occupied_now = int((timeline["chair_state"] == "Occupied").sum())
    freeing_soon = int(
        ((timeline["chair_state"] == "Occupied") & (timeline["minutes_remaining"] <= 120)).sum()
    )
    reserved_next = int((timeline["chair_state"] == "Reserved").sum())
    free_now = max(total_chairs - occupied_now - reserved_next, 0)
    total_required = occupied_now + reserved_next
    conflict_count = max(total_required - total_chairs, 0)

    metrics = {
        "total_chairs": total_chairs,
        "occupied_now": occupied_now,
        "free_now": free_now,
        "freeing_soon": freeing_soon,
        "reserved_next": reserved_next,
        "conflicts": conflict_count,
    }
    return timeline, metrics

