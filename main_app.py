from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from infusion_time_engine import INFUSION_DURATION_MAP, build_chair_allocation, regimen_duration_hours
from ml_engine import compute_ml_signals
from patient_interface_dashboard import patient_interface_dashboard


st.set_page_config(page_title="Hospital Oncology Workflow Engine", layout="wide")


GUIDELINE_NOTICE = (
    "Coordination-support system only: this platform assists workflow planning and "
    "operations but does not replace oncologist clinical judgement."
)
DATA_PATH = "sample_data/chemo_patients.csv"


def _parse_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def build_guideline_map() -> Dict[str, List[dict]]:
    return {
        "Breast": [
            {
                "name": "Breast HER2 positive",
                "condition": {"biomarker": ["HER2 positive"]},
                "sequence": [
                    {"phase": "Cycle 1-4", "regimen": "AC", "cycles": 4, "frequency_days": 21},
                    {"phase": "Cycle 5-8", "regimen": "THP", "cycles": 4, "frequency_days": 21},
                    {
                        "phase": "Maintenance",
                        "regimen": "Trastuzumab maintenance",
                        "cycles": 14,
                        "frequency_days": 21,
                    },
                ],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            },
            {
                "name": "Breast HER2 negative stage II",
                "condition": {"stage": ["II"], "biomarker": ["HER2 negative"]},
                "sequence": [
                    {"phase": "Cycle 1-4", "regimen": "Doxorubicin + Cyclophosphamide", "cycles": 4, "frequency_days": 21},
                    {"phase": "Cycle 5-8", "regimen": "Paclitaxel ± Carboplatin", "cycles": 4, "frequency_days": 21},
                ],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            },
            {
                "name": "Breast HER2 negative",
                "condition": {"biomarker": ["HER2 negative"]},
                "sequence": [
                    {"phase": "Cycle 1-4", "regimen": "AC", "cycles": 4, "frequency_days": 21},
                    {"phase": "Cycle 5-8", "regimen": "Paclitaxel", "cycles": 4, "frequency_days": 21},
                ],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            },
        ],
        "Colorectal": [
            {
                "name": "Colorectal stage III",
                "condition": {"stage": ["III"]},
                "sequence": [{"phase": "Cycle 1-8", "regimen": "FOLFOX or CAPOX", "cycles": 8, "frequency_days": 14}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Head and Neck": [
            {
                "name": "Head & Neck locally advanced",
                "condition": {"stage": ["III", "IVA", "IVB"]},
                "sequence": [{"phase": "Concurrent", "regimen": "Cisplatin + RT", "cycles": 6, "frequency_days": 7}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "DLBCL": [
            {
                "name": "DLBCL standard",
                "condition": {},
                "sequence": [{"phase": "Cycle 1-6", "regimen": "R-CHOP", "cycles": 6, "frequency_days": 21}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Hodgkin lymphoma": [
            {
                "name": "Hodgkin lymphoma standard",
                "condition": {},
                "sequence": [{"phase": "Cycle 1-6", "regimen": "ABVD", "cycles": 6, "frequency_days": 28}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Multiple myeloma": [
            {
                "name": "Multiple myeloma induction",
                "condition": {},
                "sequence": [{"phase": "Induction", "regimen": "VRd", "cycles": 4, "frequency_days": 21}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "AML": [
            {
                "name": "AML induction",
                "condition": {},
                "sequence": [{"phase": "Induction", "regimen": "7+3 induction", "cycles": 1, "frequency_days": 28}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Lung": [
            {
                "name": "Lung EGFR mutated advanced",
                "condition": {"biomarker": ["EGFR mutated"]},
                "sequence": [{"phase": "Continuous", "regimen": "Osimertinib", "cycles": 6, "frequency_days": 28}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Stomach": [
            {
                "name": "Stomach perioperative",
                "condition": {},
                "sequence": [{"phase": "Perioperative", "regimen": "FLOT", "cycles": 8, "frequency_days": 14}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Ovary": [
            {
                "name": "Ovary first-line",
                "condition": {},
                "sequence": [{"phase": "Cycle 1-6", "regimen": "Carboplatin + Paclitaxel", "cycles": 6, "frequency_days": 21}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
        "Prostate": [
            {
                "name": "Prostate metastatic",
                "condition": {},
                "sequence": [{"phase": "Cycle 1-6", "regimen": "Docetaxel + ADT", "cycles": 6, "frequency_days": 21}],
                "sources": ["NCCN", "ESMO", "NCG India", "ICMR"],
            }
        ],
    }


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def match_protocol(cancer_type: str, stage: str, biomarker: str) -> List[dict]:
    candidates = build_guideline_map().get(cancer_type, [])
    matched: List[dict] = []
    for protocol in candidates:
        condition = protocol.get("condition", {})
        stage_ok = True
        biomarker_ok = True
        if "stage" in condition:
            stage_ok = stage in condition["stage"]
        if "biomarker" in condition:
            biomarker_ok = biomarker in condition["biomarker"]
        if stage_ok and biomarker_ok:
            matched.append(protocol)
    return matched or candidates


def default_patient_df() -> pd.DataFrame:
    path = DATA_PATH
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        return df
    return pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "name": "Asha Reddy",
                "age": 49,
                "gender": "Female",
                "cancer_type": "Breast",
                "histology": "Invasive ductal carcinoma",
                "stage": "II",
                "biomarker": "HER2 negative",
                "treatment_intent": "Curative",
                "regimen_sequence": "AC->Paclitaxel",
                "cycle_number": 2,
                "last_cycle_date": "2026-04-12",
                "frequency_days": 21,
                "lab_status": "Pending",
                "pap_status": "Pending",
                "chair_hours": 2,
                "ward_assigned": "Ward A",
            }
        ]
    )


def seed_demo_patients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "name": "Asha Reddy",
                "age": 49,
                "gender": "Female",
                "cancer_type": "Breast",
                "histology": "Invasive ductal carcinoma",
                "stage": "II",
                "biomarker": "HER2 negative",
                "treatment_intent": "Curative",
                "regimen_sequence": "AC->Paclitaxel",
                "cycle_number": 2,
                "last_cycle_date": "2026-04-12",
                "frequency_days": 21,
                "lab_status": "Pending",
                "pap_status": "Pending",
                "chair_hours": 2,
                "ward_assigned": "Ward A",
            },
            {
                "patient_id": "P002",
                "name": "Ravi Kumar",
                "age": 58,
                "gender": "Male",
                "cancer_type": "DLBCL",
                "histology": "Diffuse large B-cell lymphoma",
                "stage": "III",
                "biomarker": "CD20 positive",
                "treatment_intent": "Curative",
                "regimen_sequence": "R-CHOP",
                "cycle_number": 3,
                "last_cycle_date": "2026-04-10",
                "frequency_days": 21,
                "lab_status": "Ready",
                "pap_status": "Ready",
                "chair_hours": 5,
                "ward_assigned": "Ward B",
            },
            {
                "patient_id": "P003",
                "name": "Meena Shah",
                "age": 62,
                "gender": "Female",
                "cancer_type": "Colorectal",
                "histology": "Adenocarcinoma",
                "stage": "III",
                "biomarker": "Unknown",
                "treatment_intent": "Adjuvant",
                "regimen_sequence": "FOLFOX",
                "cycle_number": 1,
                "last_cycle_date": "2026-04-17",
                "frequency_days": 14,
                "lab_status": "Pending",
                "pap_status": "Ready",
                "chair_hours": 4,
                "ward_assigned": "Ward C",
            },
            {
                "patient_id": "P004",
                "name": "Suresh Patel",
                "age": 64,
                "gender": "Male",
                "cancer_type": "Lung",
                "histology": "Adenocarcinoma",
                "stage": "IV",
                "biomarker": "EGFR mutated",
                "treatment_intent": "Palliative",
                "regimen_sequence": "Osimertinib",
                "cycle_number": 5,
                "last_cycle_date": "2026-04-01",
                "frequency_days": 28,
                "lab_status": "Ready",
                "pap_status": "Ready",
                "chair_hours": 1,
                "ward_assigned": "Daycare 2",
            },
            {
                "patient_id": "P005",
                "name": "Kavya Rao",
                "age": 31,
                "gender": "Female",
                "cancer_type": "Hodgkin lymphoma",
                "histology": "Nodular sclerosis",
                "stage": "II",
                "biomarker": "Unknown",
                "treatment_intent": "Curative",
                "regimen_sequence": "ABVD",
                "cycle_number": 4,
                "last_cycle_date": "2026-04-05",
                "frequency_days": 28,
                "lab_status": "Ready",
                "pap_status": "Pending",
                "chair_hours": 3,
                "ward_assigned": "Ward A",
            },
            {
                "patient_id": "P006",
                "name": "Arjun Menon",
                "age": 55,
                "gender": "Male",
                "cancer_type": "Head and Neck",
                "histology": "Squamous cell carcinoma",
                "stage": "IVA",
                "biomarker": "Unknown",
                "treatment_intent": "Curative",
                "regimen_sequence": "Cisplatin + RT",
                "cycle_number": 2,
                "last_cycle_date": "2026-04-15",
                "frequency_days": 7,
                "lab_status": "Pending",
                "pap_status": "Pending",
                "chair_hours": 3,
                "ward_assigned": "Ward D",
            },
            {
                "patient_id": "P007",
                "name": "Lakshmi Iyer",
                "age": 47,
                "gender": "Female",
                "cancer_type": "Ovary",
                "histology": "High-grade serous carcinoma",
                "stage": "III",
                "biomarker": "Unknown",
                "treatment_intent": "Adjuvant",
                "regimen_sequence": "Carboplatin + Paclitaxel",
                "cycle_number": 2,
                "last_cycle_date": "2026-04-08",
                "frequency_days": 21,
                "lab_status": "Ready",
                "pap_status": "Ready",
                "chair_hours": 4,
                "ward_assigned": "Ward C",
            },
            {
                "patient_id": "P008",
                "name": "Naveen Reddy",
                "age": 66,
                "gender": "Male",
                "cancer_type": "Prostate",
                "histology": "Adenocarcinoma",
                "stage": "IV",
                "biomarker": "Unknown",
                "treatment_intent": "Palliative",
                "regimen_sequence": "Docetaxel + ADT",
                "cycle_number": 4,
                "last_cycle_date": "2026-04-03",
                "frequency_days": 21,
                "lab_status": "Ready",
                "pap_status": "Ready",
                "chair_hours": 3,
                "ward_assigned": "Ward B",
            },
            {
                "patient_id": "P009",
                "name": "Farah Khan",
                "age": 39,
                "gender": "Female",
                "cancer_type": "Breast",
                "histology": "Invasive ductal carcinoma",
                "stage": "III",
                "biomarker": "HER2 positive",
                "treatment_intent": "Neoadjuvant",
                "regimen_sequence": "AC -> THP -> Trastuzumab maintenance",
                "cycle_number": 5,
                "last_cycle_date": "2026-04-16",
                "frequency_days": 21,
                "lab_status": "Pending",
                "pap_status": "Pending",
                "chair_hours": 6,
                "ward_assigned": "Ward A",
            },
            {
                "patient_id": "P010",
                "name": "Joseph Mathew",
                "age": 51,
                "gender": "Male",
                "cancer_type": "Multiple myeloma",
                "histology": "Plasma cell neoplasm",
                "stage": "III",
                "biomarker": "Unknown",
                "treatment_intent": "Curative",
                "regimen_sequence": "VRd",
                "cycle_number": 3,
                "last_cycle_date": "2026-04-11",
                "frequency_days": 21,
                "lab_status": "Ready",
                "pap_status": "Pending",
                "chair_hours": 4,
                "ward_assigned": "Daycare 1",
            },
        ]
    )


def persist_dataset(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False)


def protocol_to_regimen_sequence(plan_rows: List[dict]) -> str:
    if not plan_rows:
        return ""
    return " -> ".join(str(p.get("regimen", "")).strip() for p in plan_rows if str(p.get("regimen", "")).strip())


def protocol_to_frequency(plan_rows: List[dict], default_frequency: int = 21) -> int:
    if not plan_rows:
        return default_frequency
    return _as_int(plan_rows[0].get("frequency_days"), default_frequency)


def protocol_total_cycles(plan_rows: List[dict], default_cycles: int = 1) -> int:
    if not plan_rows:
        return default_cycles
    total = 0
    for phase in plan_rows:
        total += _as_int(phase.get("cycles"), 0)
    return total if total > 0 else default_cycles


def infer_infusion_hours(regimen_text: str, fallback: float = 3.0) -> float:
    return regimen_duration_hours(regimen_text, fallback=fallback)


def compute_schedule(last_cycle_date: object, frequency_days: int) -> dict:
    last_dt = _parse_date(last_cycle_date) or date.today()
    next_cycle_date = last_dt + timedelta(days=int(frequency_days))
    pap_date = next_cycle_date - timedelta(days=2)
    lab_date = pap_date - timedelta(days=1)
    return {
        "last_cycle_date": last_dt,
        "next_cycle_date": next_cycle_date,
        "pap_date": pap_date,
        "lab_date": lab_date,
    }


def readiness_status(lab_status: str, pap_status: str) -> str:
    if str(lab_status).lower() == "ready" and str(pap_status).lower() == "ready":
        return "Ready"
    if str(lab_status).lower() != "ready":
        return "Pending labs"
    if str(pap_status).lower() != "ready":
        return "Pending PAP"
    return "Delayed"


def build_patient_timeline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        schedule = compute_schedule(row.get("last_cycle_date"), _as_int(row.get("frequency_days", 21), 21))
        ready = readiness_status(str(row.get("lab_status", "")), str(row.get("pap_status", "")))
        rows.append(
            {
                "patient_id": row.get("patient_id"),
                "name": row.get("name"),
                "cycle_number": row.get("cycle_number"),
                "regimen_sequence": row.get("regimen_sequence"),
                "lab_date": schedule["lab_date"],
                "pap_date": schedule["pap_date"],
                "next_cycle_date": schedule["next_cycle_date"],
                "readiness": ready,
                "chair_hours": row.get("chair_hours", infer_infusion_hours(str(row.get("regimen_sequence", "")))),
            }
        )
    return pd.DataFrame(rows)


def apply_safe_rescheduling(timeline_df: pd.DataFrame) -> pd.DataFrame:
    reason_map = {
        "Pending labs": "neutropenia",
        "Pending PAP": "platelet low",
        "Delayed": "patient unavailable",
    }
    output = timeline_df.copy()
    output["safe_window_date"] = output["next_cycle_date"]
    output["delay_reason"] = ""
    for i, row in output.iterrows():
        ready = row["readiness"]
        if ready != "Ready":
            output.at[i, "safe_window_date"] = row["next_cycle_date"] + timedelta(days=7)
            output.at[i, "delay_reason"] = reason_map.get(ready, "creatinine elevated")
    return output


def lab_test_requirements(regimen_sequence: str) -> List[str]:
    tests = ["CBC", "LFT", "RFT"]
    name = regimen_sequence.lower()
    if "trastuzumab" in name or "thp" in name:
        tests.append("ECHO")
    if "cisplatin" in name:
        tests.append("Creatinine clearance")
    return tests


def ensure_state() -> None:
    if "patients_df" not in st.session_state:
        st.session_state.patients_df = default_patient_df()
    if "confirmed_plan" not in st.session_state:
        st.session_state.confirmed_plan = []
    if "complaints" not in st.session_state:
        st.session_state.complaints = []
    if "reminders" not in st.session_state:
        st.session_state.reminders = []
    if "integration_events" not in st.session_state:
        st.session_state.integration_events = []
    if "ops_alerts" not in st.session_state:
        st.session_state.ops_alerts = []
    if "notifications" not in st.session_state:
        st.session_state.notifications = []


def oncologist_dashboard() -> None:
    st.header("Oncologist Protocol Decision Dashboard")
    st.info(GUIDELINE_NOTICE)
    st.caption("Guideline base: NCCN, ESMO, NCG India, ICMR.")

    patient_df = st.session_state.patients_df.copy()
    patient_options = [f"{r.patient_id} | {r.name}" for r in patient_df.itertuples()]
    selected_patient_label = st.selectbox("Select patient to plan", patient_options)
    selected_patient_id = selected_patient_label.split("|")[0].strip()
    selected_row = patient_df[patient_df["patient_id"] == selected_patient_id].iloc[0]
    base_timeline = apply_safe_rescheduling(build_patient_timeline(patient_df.copy()))
    ml_scored, _ = compute_ml_signals(patient_df, base_timeline)
    scored_row = ml_scored[ml_scored["patient_id"] == selected_patient_id].iloc[0]
    st.subheader("Workflow Risk Insight (Support Only)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Cycle delay likelihood", f"{scored_row['cycle_delay_risk_pct']}%")
    r2.metric("Attendance risk", f"{scored_row['no_show_risk_pct']}%")
    r3.metric("Priority level", str(scored_row["workflow_priority"]))
    st.caption(str(scored_row["workflow_recommendation"]))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        cancer_type = st.selectbox(
            "Cancer type",
            [
                "Breast",
                "Head and Neck",
                "Colorectal",
                "Lung",
                "Stomach",
                "Ovary",
                "Prostate",
                "DLBCL",
                "Hodgkin lymphoma",
                "Multiple myeloma",
                "AML",
            ],
            index=[
                "Breast",
                "Head and Neck",
                "Colorectal",
                "Lung",
                "Stomach",
                "Ovary",
                "Prostate",
                "DLBCL",
                "Hodgkin lymphoma",
                "Multiple myeloma",
                "AML",
            ].index(str(selected_row.get("cancer_type", "Breast")) if str(selected_row.get("cancer_type", "Breast")) in [
                "Breast",
                "Head and Neck",
                "Colorectal",
                "Lung",
                "Stomach",
                "Ovary",
                "Prostate",
                "DLBCL",
                "Hodgkin lymphoma",
                "Multiple myeloma",
                "AML",
            ] else "Breast"),
        )
    with c2:
        histology = st.text_input("Histology subtype", str(selected_row.get("histology", "Adenocarcinoma")))
    with c3:
        stage_choices = ["I", "II", "III", "IVA", "IVB", "IV"]
        selected_stage = str(selected_row.get("stage", "II"))
        stage = st.selectbox("Stage", stage_choices, index=stage_choices.index(selected_stage) if selected_stage in stage_choices else 0)
    with c4:
        biomarker_choices = ["HER2 negative", "HER2 positive", "EGFR mutated", "CD20 positive", "Unknown"]
        selected_biomarker = str(selected_row.get("biomarker", "Unknown"))
        biomarker = st.selectbox(
            "Biomarker status",
            biomarker_choices,
            index=biomarker_choices.index(selected_biomarker) if selected_biomarker in biomarker_choices else 4,
        )
    with c5:
        intent_choices = ["Curative", "Neoadjuvant", "Adjuvant", "Palliative", "Maintenance"]
        selected_intent = str(selected_row.get("treatment_intent", "Curative"))
        intent = st.selectbox("Treatment intent", intent_choices, index=intent_choices.index(selected_intent) if selected_intent in intent_choices else 0)

    protocols = match_protocol(cancer_type, stage, biomarker)
    if protocols:
        st.subheader("Suggested guideline protocol sequences")
        protocol_names = [p["name"] for p in protocols]
        chosen_name = st.selectbox("Suggested protocol", protocol_names)
        selected = next(p for p in protocols if p["name"] == chosen_name)
        st.write("Guideline source:", ", ".join(selected["sources"]))
    else:
        selected = {"sequence": []}

    st.subheader("Editable treatment timeline")
    plan_rows = []
    for i, phase in enumerate(selected.get("sequence", [])):
        cols = st.columns([2, 3, 1, 1, 2])
        with cols[0]:
            phase_name = st.text_input(f"Phase {i + 1}", value=phase["phase"], key=f"phase_{i}")
        with cols[1]:
            regimen = st.text_input(f"Regimen {i + 1}", value=phase["regimen"], key=f"regimen_{i}")
        with cols[2]:
            cycles = st.number_input(f"Cycles {i + 1}", min_value=1, value=int(phase["cycles"]), key=f"cycles_{i}")
        with cols[3]:
            freq = st.number_input(
                f"Frequency days {i + 1}",
                min_value=1,
                value=int(phase["frequency_days"]),
                key=f"freq_{i}",
            )
        with cols[4]:
            targeted = st.checkbox(f"Add targeted therapy phase {i + 1}", value=False, key=f"targeted_{i}")
        regimen_final = regimen + (" + targeted therapy" if targeted else "")
        plan_rows.append({"phase": phase_name, "regimen": regimen_final, "cycles": int(cycles), "frequency_days": int(freq)})

    add_maint = st.checkbox("Add maintenance therapy")
    if add_maint:
        maint_regimen = st.text_input("Maintenance regimen", "Trastuzumab maintenance")
        maint_cycles = st.number_input("Maintenance cycles", min_value=1, value=14)
        maint_freq = st.number_input("Maintenance frequency days", min_value=1, value=21)
        plan_rows.append(
            {
                "phase": "Maintenance",
                "regimen": maint_regimen,
                "cycles": int(maint_cycles),
                "frequency_days": int(maint_freq),
            }
        )

    if plan_rows:
        st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)
        start_date = _parse_date(selected_row.get("last_cycle_date")) or date.today()
        timeline_vis = []
        cursor = start_date
        for phase in plan_rows:
            phase_cycles = _as_int(phase.get("cycles"), 1)
            phase_freq = _as_int(phase.get("frequency_days"), 21)
            phase_start = cursor + timedelta(days=1)
            phase_end = phase_start + timedelta(days=(phase_cycles * phase_freq))
            timeline_vis.append(
                {
                    "phase": phase["phase"],
                    "regimen": phase["regimen"],
                    "start": phase_start,
                    "end": phase_end,
                    "cycles": phase_cycles,
                }
            )
            cursor = phase_end
        st.subheader("Protocol timeline view")
        st.dataframe(pd.DataFrame(timeline_vis), use_container_width=True)

    if st.button("Finalize treatment plan"):
        st.session_state.confirmed_plan = plan_rows
        updated_df = st.session_state.patients_df.copy()
        idx = updated_df.index[updated_df["patient_id"] == selected_patient_id][0]
        sequence = protocol_to_regimen_sequence(plan_rows)
        freq_days = protocol_to_frequency(plan_rows, _as_int(updated_df.at[idx, "frequency_days"], 21))
        total_cycles = protocol_total_cycles(plan_rows, _as_int(updated_df.at[idx, "cycle_number"], 1))
        updated_df.at[idx, "cancer_type"] = cancer_type
        updated_df.at[idx, "histology"] = histology
        updated_df.at[idx, "stage"] = stage
        updated_df.at[idx, "biomarker"] = biomarker
        updated_df.at[idx, "treatment_intent"] = intent
        updated_df.at[idx, "regimen_sequence"] = sequence
        updated_df.at[idx, "frequency_days"] = freq_days
        updated_df.at[idx, "cycle_number"] = total_cycles
        updated_df.at[idx, "chair_hours"] = infer_infusion_hours(sequence)
        st.session_state.patients_df = updated_df
        schedule = compute_schedule(updated_df.at[idx, "last_cycle_date"], freq_days)
        tests = ", ".join(lab_test_requirements(sequence))
        patient_msg = (
            f"{selected_patient_id}: treatment finalized. Admission around {schedule['next_cycle_date']}, "
            f"PAP on {schedule['pap_date']}, labs on {schedule['lab_date']}."
        )
        lab_msg = (
            f"{selected_patient_id}: collect samples by {schedule['lab_date']} and run tests ({tests}) "
            f"for planned cycle readiness."
        )
        pharmacy_msg = (
            f"{selected_patient_id}: provisional regimen planned ({sequence}). "
            "Prepare pharmacy forecast; final release after readiness confirmation."
        )
        daycare_msg = (
            f"{selected_patient_id}: provisional slot planning started with cycle frequency {freq_days} days."
        )
        st.session_state.notifications.append({"time": str(datetime.now()), "target": "Patient", "message": patient_msg})
        st.session_state.notifications.append({"time": str(datetime.now()), "target": "Lab technician", "message": lab_msg})
        st.session_state.notifications.append({"time": str(datetime.now()), "target": "Pharmacy dashboard", "message": pharmacy_msg})
        st.session_state.notifications.append({"time": str(datetime.now()), "target": "Daycare dashboard", "message": daycare_msg})
        st.success(
            f"Doctor finalized plan for {selected_patient_id}. "
            "Oncologist plan is now the primary driver for all downstream dashboards."
        )


def patient_timeline_panel() -> pd.DataFrame:
    st.header("Patient Timeline")
    df = st.session_state.patients_df.copy()
    timeline = build_patient_timeline(df)
    timeline = apply_safe_rescheduling(timeline)
    st.dataframe(timeline, use_container_width=True)
    return timeline


def daycare_dashboard(timeline: pd.DataFrame, ml_scores: pd.DataFrame) -> None:
    st.header("Daycare Workflow Dashboard")
    today = date.today()
    timeline["next_cycle_date"] = pd.to_datetime(timeline["next_cycle_date"]).dt.date
    today_df = timeline[timeline["next_cycle_date"] == today]
    ready_df = timeline[timeline["readiness"] == "Ready"]
    pending_labs = timeline[timeline["readiness"] == "Pending labs"]
    pending_pap = timeline[timeline["readiness"] == "Pending PAP"]
    rescheduled = timeline[timeline["safe_window_date"] > timeline["next_cycle_date"]]

    total_chairs = 20.0
    today_hours = float(today_df["chair_hours"].astype(float).sum()) if not today_df.empty else 0.0
    chair_util = min(100.0, (today_hours / (total_chairs * 8.0)) * 100.0)
    tomorrow = today + timedelta(days=1)
    tomorrow_count = int((timeline["next_cycle_date"] == tomorrow).sum())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Today chemo patients", int(len(today_df)))
    m2.metric("Ready patients", int(len(ready_df)))
    m3.metric("Pending labs", int(len(pending_labs)))
    m4.metric("Pending PAP", int(len(pending_pap)))
    m5.metric("Rescheduled patients", int(len(rescheduled)))
    m6.metric("Chair utilization %", f"{chair_util:.1f}%")
    st.caption(f"Tomorrow workload prediction: {tomorrow_count} patients")
    st.subheader("Infusion-slot allocation timeline")
    st.dataframe(timeline[["patient_id", "name", "regimen_sequence", "chair_hours", "next_cycle_date", "safe_window_date"]], use_container_width=True)
    slot_df = timeline.copy()
    slot_df["slot_start"] = pd.to_datetime(slot_df["safe_window_date"]).dt.strftime("%Y-%m-%d") + " 09:00"
    slot_df["slot_end"] = (
        pd.to_datetime(slot_df["safe_window_date"]) + pd.to_timedelta(slot_df["chair_hours"], unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M")
    st.subheader("Tomorrow slot forecast")
    st.dataframe(slot_df[["patient_id", "name", "slot_start", "slot_end", "chair_hours"]], use_container_width=True)
    st.subheader("Ward preparation alerts")
    if st.session_state.ops_alerts:
        st.dataframe(pd.DataFrame(st.session_state.ops_alerts), use_container_width=True)
    else:
        st.info("No active operations alerts.")
    st.subheader("Risk watchlist for coordination")
    risk_cols = ["patient_id", "name", "cycle_delay_risk_pct", "no_show_risk_pct", "workflow_priority", "workflow_recommendation"]
    risk_df = ml_scores.sort_values("cycle_delay_risk", ascending=False)[risk_cols].head(8).rename(
        columns={
            "patient_id": "Patient ID",
            "name": "Name",
            "cycle_delay_risk_pct": "Cycle delay likelihood (%)",
            "no_show_risk_pct": "Attendance risk (%)",
            "workflow_priority": "Priority level",
            "workflow_recommendation": "Recommended coordination action",
        }
    )
    st.dataframe(risk_df, use_container_width=True)


def pharmacy_dashboard(timeline: pd.DataFrame, ml_scores: pd.DataFrame) -> None:
    st.header("Pharmacy Coordination Dashboard")
    df = st.session_state.patients_df.copy()
    merged = timeline.merge(df[["patient_id", "ward_assigned"]], on="patient_id", how="left")
    now = date.today()
    in_24 = merged[pd.to_datetime(merged["next_cycle_date"]).dt.date <= now + timedelta(days=1)]
    in_48 = merged[pd.to_datetime(merged["next_cycle_date"]).dt.date <= now + timedelta(days=2)]
    prep = in_24[in_24["ward_assigned"].fillna("").astype(str).str.strip() != ""]
    grouped = in_48.groupby("regimen_sequence", as_index=False).agg(patients=("patient_id", "count"))

    st.write("When ward is assigned, pharmacy prep alert is triggered.")
    st.subheader("Next 24-hour preparation list")
    st.dataframe(prep[["patient_id", "name", "regimen_sequence", "ward_assigned", "next_cycle_date"]], use_container_width=True)
    st.subheader("Next 48-hour forecast")
    st.dataframe(in_48[["patient_id", "name", "regimen_sequence", "next_cycle_date"]], use_container_width=True)
    st.subheader("Grouped identical regimens (wastage reduction)")
    st.dataframe(grouped, use_container_width=True)
    st.subheader("Operations-triggered prep alerts")
    if st.session_state.ops_alerts:
        st.dataframe(pd.DataFrame(st.session_state.ops_alerts), use_container_width=True)
    else:
        st.info("No chair-freeing alerts received yet.")
    st.subheader("Lab coordinator notifications")
    pharmacy_notes = [n for n in st.session_state.notifications if n.get("target") == "Pharmacy dashboard"]
    if pharmacy_notes:
        st.dataframe(pd.DataFrame(pharmacy_notes), use_container_width=True)
    else:
        st.info("No lab-result-driven pharmacy alerts yet.")
    st.subheader("Predicted high-risk delays (prep caution)")
    high_risk = ml_scores[ml_scores["cycle_delay_risk"] >= 0.6][["patient_id", "name", "cycle_delay_risk_pct", "workflow_recommendation"]].rename(
        columns={
            "patient_id": "Patient ID",
            "name": "Name",
            "cycle_delay_risk_pct": "Cycle delay likelihood (%)",
            "workflow_recommendation": "Recommended coordination action",
        }
    )
    if not high_risk.empty:
        st.dataframe(high_risk, use_container_width=True)
    else:
        st.info("No high delay-risk cases currently.")


def build_lab_collection_tasks(timeline: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    today = date.today()
    df = st.session_state.patients_df.copy()
    for _, row in timeline.iterrows():
        patient_meta = df[df["patient_id"] == row["patient_id"]].iloc[0]
        lab_date = _parse_date(row["lab_date"]) or today
        sample_mode = "Home collection" if int(str(row["patient_id"]).replace("P", "")) % 2 == 1 else "Hospital walk-in"
        urgency = "High" if lab_date <= today + timedelta(days=1) else "Routine"
        tasks.append(
            {
                "patient_id": row["patient_id"],
                "name": row["name"],
                "collection_date": lab_date,
                "sample_mode": sample_mode,
                "required_tests": ", ".join(lab_test_requirements(str(row["regimen_sequence"]))),
                "urgency": urgency,
                "current_lab_status": patient_meta.get("lab_status", "Pending"),
            }
        )
    return pd.DataFrame(tasks)


def lab_coordinator_dashboard(timeline: pd.DataFrame, ml_scores: pd.DataFrame) -> None:
    st.header("Lab Coordinator Dashboard")
    st.caption("Home sample collection coordination, report upload, and readiness notifications.")

    tasks_df = build_lab_collection_tasks(timeline)
    st.subheader("Upcoming blood sample collection tasks")
    st.dataframe(tasks_df, use_container_width=True)
    home_due = tasks_df[(tasks_df["sample_mode"] == "Home collection") & (tasks_df["urgency"] == "High")]
    if not home_due.empty:
        st.warning(f"Home blood collection alerts: {len(home_due)} high-priority collections due soon.")
    high_risk_ids = set(ml_scores[ml_scores["cycle_delay_risk"] >= 0.6]["patient_id"].astype(str).tolist())
    if high_risk_ids:
        st.warning(f"Priority alert: {len(high_risk_ids)} patients are high-risk for delay; prioritize sample collection.")

    st.subheader("Upload lab report and trigger coordination flow")
    patient_options = [f"{r.patient_id} | {r.name}" for r in st.session_state.patients_df.itertuples()]
    selected_patient_label = st.selectbox("Select patient report", patient_options, key="lab_patient")
    selected_patient_id = selected_patient_label.split("|")[0].strip()
    report_result = st.selectbox("Lab report summary", ["Values normal", "Values abnormal"], key="lab_result")
    counsellor_role = st.selectbox(
        "If abnormal, assign medication counselling to",
        ["Clinical Pharmacist", "DMO", "Fellowship Doctor", "DNBs"],
        key="counsellor_role",
    )
    if st.button("Upload report and notify teams"):
        patients = st.session_state.patients_df.copy()
        idx = patients.index[patients["patient_id"] == selected_patient_id][0]
        if report_result == "Values normal":
            patients.at[idx, "lab_status"] = "Ready"
            patients.at[idx, "pap_status"] = "Ready"
            msg = f"{selected_patient_id}: labs acceptable. Patient can proceed for treatment as planned."
            st.session_state.notifications.append({"time": str(datetime.now()), "target": "Patient", "message": msg})
            st.session_state.notifications.append({"time": str(datetime.now()), "target": "Pharmacy dashboard", "message": msg})
            st.toast("Lab uploaded: patient + pharmacy notified for treatment readiness.")
        else:
            patients.at[idx, "lab_status"] = "Pending"
            patients.at[idx, "pap_status"] = "Pending"
            msg = f"{selected_patient_id}: abnormal labs. Chemo reschedule suggested; follow-up medication counselling assigned to {counsellor_role}."
            st.session_state.notifications.append({"time": str(datetime.now()), "target": "Patient", "message": msg})
            st.session_state.notifications.append({"time": str(datetime.now()), "target": "Pharmacy dashboard", "message": msg})
            st.session_state.notifications.append({"time": str(datetime.now()), "target": "Daycare dashboard", "message": msg})
            st.toast("Abnormal report uploaded: reschedule workflow + counselling assignment triggered.")
        st.session_state.patients_df = patients
        st.success("Lab coordination updates pushed to connected dashboards.")

    st.subheader("Recent coordination notifications")
    if st.session_state.notifications:
        st.dataframe(pd.DataFrame(st.session_state.notifications), use_container_width=True)
    else:
        st.info("No notifications generated yet.")

    st.subheader("Pending instructions for lab technicians")
    lab_notes = [n for n in st.session_state.notifications if n.get("target") == "Lab technician"]
    if lab_notes:
        st.dataframe(pd.DataFrame(lab_notes), use_container_width=True)
    else:
        st.info("No pending technician instructions.")


def operations_dashboard(timeline: pd.DataFrame, ml_forecast: Dict[str, float]) -> None:
    st.header("Operations Dashboard")
    st.caption("Real-time infusion bed/chair utilization and vacancy prediction for daycare operations.")

    total_chairs = st.number_input("Total infusion chairs available", min_value=1, value=20)
    allocation_df, metrics = build_chair_allocation(timeline, total_chairs=int(total_chairs))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total infusion chairs", metrics["total_chairs"])
    m2.metric("Occupied chairs", metrics["occupied_now"])
    m3.metric("Chairs free now", metrics["free_now"])
    m4.metric("Chairs freeing soon", metrics["freeing_soon"])
    m5.metric("Reserved for next patients", metrics["reserved_next"])

    c1, c2, c3 = st.columns(3)
    c1.success("Green = available")
    c2.warning("Yellow = freeing soon")
    c3.error("Red = occupied")

    st.subheader("Infusion duration mapping table")
    mapping_df = pd.DataFrame(
        [{"regimen": reg, "duration_hours": hrs, "notes": "No chair block" if hrs == 0 else "Chair blocked"} for reg, hrs in INFUSION_DURATION_MAP.items()]
    )
    st.dataframe(mapping_df, use_container_width=True)

    def state_color(row: pd.Series) -> str:
        state = row["chair_state"]
        if state == "Completed":
            return "Available"
        if state == "Occupied" and row["minutes_remaining"] <= 120:
            return "Freeing soon"
        if state == "Occupied":
            return "Occupied"
        return "Reserved"

    allocation_df["visual_state"] = allocation_df.apply(state_color, axis=1)
    allocation_df["release_time"] = pd.to_datetime(allocation_df["slot_end"]).dt.strftime("%Y-%m-%d %H:%M")
    allocation_df["predicted_vacancy"] = allocation_df.apply(
        lambda r: r["free_prediction"] if r["chair_state"] == "Occupied" else ("Reserved for next patient" if r["chair_state"] == "Reserved" else "Free now"),
        axis=1,
    )

    st.subheader("Operations bed utilization engine")
    st.dataframe(
        allocation_df[
            [
                "chair_id",
                "patient_id",
                "name",
                "regimen_sequence",
                "chair_hours",
                "chair_state",
                "visual_state",
                "release_time",
                "predicted_vacancy",
                "reservation_status",
            ]
        ],
        use_container_width=True,
    )
    st.subheader("Gantt-style chair occupancy timeline")
    gantt_df = allocation_df.copy()
    gantt_df["slot_start"] = pd.to_datetime(gantt_df["slot_start"])
    gantt_df["slot_end"] = pd.to_datetime(gantt_df["slot_end"])
    gantt_df["chair_label"] = gantt_df["chair_id"].apply(lambda x: f"Chair {int(x)}")
    gantt_df["tooltip_label"] = gantt_df.apply(
        lambda r: f"{r['patient_id']} | {r['name']} | {r['regimen_sequence']}",
        axis=1,
    )
    color_scale = alt.Scale(
        domain=["Available", "Freeing soon", "Occupied", "Reserved"],
        range=["#2ca02c", "#ffbf00", "#d62728", "#1f77b4"],
    )
    gantt_chart = (
        alt.Chart(gantt_df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("slot_start:T", title="Start time"),
            x2="slot_end:T",
            y=alt.Y("chair_label:N", sort="ascending", title="Infusion chairs"),
            color=alt.Color("visual_state:N", scale=color_scale, title="Chair state"),
            tooltip=[
                alt.Tooltip("tooltip_label:N", title="Patient"),
                alt.Tooltip("slot_start:T", title="Start"),
                alt.Tooltip("slot_end:T", title="End"),
                alt.Tooltip("chair_hours:Q", title="Duration (h)"),
                alt.Tooltip("predicted_vacancy:N", title="Predicted vacancy"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(gantt_chart, use_container_width=True)

    conflict_flag = metrics["conflicts"] > 0
    if conflict_flag:
        st.error(f"Bed blocking conflict detected: {metrics['conflicts']} overlapping reservations beyond available chairs.")
    else:
        st.success("No double-booking conflicts detected in current reservations.")

    soon_df = allocation_df[
        (allocation_df["chair_state"] == "Occupied") & (allocation_df["minutes_remaining"] <= 120)
    ][["chair_id", "patient_id", "name", "predicted_vacancy", "release_time"]]
    st.subheader("Predicted vacancy engine")
    if not soon_df.empty:
        st.dataframe(soon_df, use_container_width=True)
    else:
        st.info("No chairs are nearing release in the next 2 hours.")

    alerts = []
    for _, row in soon_df.iterrows():
        alerts.append(
            {
                "timestamp": str(datetime.now()),
                "target": "Daycare dashboard",
                "message": f"Chair {int(row['chair_id'])} for {row['patient_id']} {row['predicted_vacancy']} - prepare next patient admission.",
            }
        )
        alerts.append(
            {
                "timestamp": str(datetime.now()),
                "target": "Pharmacy dashboard",
                "message": f"Upcoming chair release ({row['predicted_vacancy']}) for {row['patient_id']}. Prepare next regimen.",
            }
        )
        alerts.append(
            {
                "timestamp": str(datetime.now()),
                "target": "Admission coordination layer",
                "message": f"Initiate admission workflow: chair {int(row['chair_id'])} expected free at {row['release_time']}.",
            }
        )
    st.session_state.ops_alerts = alerts

    st.subheader("Tomorrow resource forecast")
    tomorrow = date.today() + timedelta(days=1)
    alloc = allocation_df.copy()
    alloc["safe_window_date_only"] = pd.to_datetime(alloc["safe_window_date"]).dt.date
    tomorrow_df = alloc[alloc["safe_window_date_only"] == tomorrow]
    expected_patients = int(len(tomorrow_df))
    required_hours = float(tomorrow_df["chair_hours"].sum()) if not tomorrow_df.empty else 0.0
    morning = float(tomorrow_df[tomorrow_df["chair_hours"] <= 3]["chair_hours"].sum()) if not tomorrow_df.empty else 0.0
    afternoon = float(tomorrow_df[(tomorrow_df["chair_hours"] > 3) & (tomorrow_df["chair_hours"] <= 5)]["chair_hours"].sum()) if not tomorrow_df.empty else 0.0
    evening = float(tomorrow_df[tomorrow_df["chair_hours"] > 5]["chair_hours"].sum()) if not tomorrow_df.empty else 0.0
    peak_label = "Morning"
    peak_hours = morning
    if afternoon > peak_hours:
        peak_label, peak_hours = "Afternoon", afternoon
    if evening > peak_hours:
        peak_label, peak_hours = "Evening", evening
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Expected chemo patients", expected_patients)
    f2.metric("Required chair-hours", f"{required_hours:.1f}")
    f3.metric("Peak load window", peak_label)
    f4.metric("Peak window chair-hours", f"{peak_hours:.1f}")
    st.write(
        {
            "morning_load_chair_hours": round(morning, 1),
            "afternoon_load_chair_hours": round(afternoon, 1),
            "evening_load_chair_hours": round(evening, 1),
        }
    )
    st.subheader("Enhanced demand forecast")
    a1, a2 = st.columns(2)
    a1.metric("Expected next-day patients (predictive)", ml_forecast["expected_next_day_patients"])
    a2.metric("Expected next-day chair-hours (predictive)", ml_forecast["expected_next_day_chair_hours"])
    st.caption("Predictive outputs are decision-support only; oncologist and care teams remain final decision-makers.")


def admin_dashboard(timeline: pd.DataFrame, ml_scores: pd.DataFrame) -> None:
    st.header("Operations & Admin Dashboard")
    st.subheader("Lab Alert Coordination System")
    alerts = []
    for _, row in timeline.iterrows():
        tests = lab_test_requirements(str(row["regimen_sequence"]))
        urgency = "High" if row["readiness"] != "Ready" else "Routine"
        alerts.append(
            {
                "patient_id": row["patient_id"],
                "tests_required": ", ".join(tests),
                "collection_date": row["lab_date"],
                "urgency": urgency,
                "result_status": "Available" if row["readiness"] == "Ready" else "Awaited",
            }
        )
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)

    st.subheader("HIS / EMR / EHR Integration Ready Layer")
    st.code(
        "simulate_achala_interface(payload)\n"
        "simulate_his_connector(event)\n"
        "simulate_emr_ehr_sync(patient_record)",
        language="python",
    )
    if st.button("Simulate data exchange"):
        st.session_state.integration_events.append(
            {
                "event_time": str(datetime.now()),
                "connector": "Achala/HIS/EMR-EHR",
                "status": "SUCCESS",
                "payload": f"patients={len(timeline)}",
            }
        )
        st.success("Simulated: Achala, HIS, and EMR/EHR synchronization event dispatched.")
    if st.session_state.integration_events:
        st.dataframe(pd.DataFrame(st.session_state.integration_events), use_container_width=True)

    st.subheader("Complaint Escalation System")
    severity = st.selectbox("Severity", ["Low", "Medium", "High"])
    complaint_text = st.text_input("Complaint details")
    if st.button("Submit complaint"):
        if complaint_text.strip():
            st.session_state.complaints.append({"severity": severity, "detail": complaint_text, "date": str(date.today())})
            st.success("Complaint escalated to admin dashboard.")
        else:
            st.warning("Enter complaint details before submission.")
    if st.session_state.complaints:
        st.dataframe(pd.DataFrame(st.session_state.complaints), use_container_width=True)
    st.subheader("Cross-department coordination notifications")
    if st.session_state.notifications:
        st.dataframe(pd.DataFrame(st.session_state.notifications), use_container_width=True)
    else:
        st.info("No coordination notifications available.")
    st.subheader("Risk governance monitor")
    governance_df = ml_scores[
        ["patient_id", "name", "cycle_delay_risk_pct", "no_show_risk_pct", "workflow_priority", "workflow_recommendation"]
    ].rename(
        columns={
            "patient_id": "Patient ID",
            "name": "Name",
            "cycle_delay_risk_pct": "Cycle delay likelihood (%)",
            "no_show_risk_pct": "Attendance risk (%)",
            "workflow_priority": "Priority level",
            "workflow_recommendation": "Recommended coordination action",
        }
    )
    st.dataframe(governance_df, use_container_width=True)


def reminder_panel() -> None:
    st.header("Patient Reminder Panel")
    reminder_type = st.selectbox("Reminder type", ["Admission reminder", "Lab reminder", "Review reminder", "Cycle reminder"])
    patient_options = list(st.session_state.patients_df["patient_id"].astype(str).unique())
    patient_id = st.selectbox("Patient ID for reminder", patient_options)
    if st.button("Send reminder"):
        if patient_id.strip():
            st.session_state.reminders.append(
                {"patient_id": patient_id.strip(), "reminder_type": reminder_type, "status": "Sent", "sent_at": str(datetime.now())}
            )
            st.success("Reminder simulated and status updated.")
        else:
            st.warning("Enter a patient ID.")
    if st.session_state.reminders:
        st.dataframe(pd.DataFrame(st.session_state.reminders), use_container_width=True)
    st.subheader("Patient notifications from lab coordination")
    patient_notes = [n for n in st.session_state.notifications if n.get("target") == "Patient"]
    if patient_notes:
        st.dataframe(pd.DataFrame(patient_notes), use_container_width=True)
    else:
        st.info("No patient lab notifications yet.")


def complaint_panel() -> None:
    st.header("Complaint Escalation Panel")
    if st.session_state.complaints:
        df = pd.DataFrame(st.session_state.complaints)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No complaints logged.")


def main() -> None:
    ensure_state()
    st.title("Hospital-Grade Oncology Workflow Engine")
    st.warning(GUIDELINE_NOTICE)

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Oncologist Dashboard",
            "Lab Coordinator Dashboard",
            "Daycare Dashboard",
            "Pharmacy Dashboard",
            "Operations & Admin Dashboard",
            "Patient Timeline",
            "Patient Reminder Panel",
            "Patient Companion Interface",
            "Complaint Escalation Panel",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Clinical safety: coordination-support only.")
    st.sidebar.subheader("Prototype Controls")
    if st.sidebar.button("Save current dataset"):
        persist_dataset(st.session_state.patients_df)
        st.sidebar.success("Dataset saved to sample_data/chemo_patients.csv")
    if st.sidebar.button("Reset to demo dataset"):
        st.session_state.patients_df = seed_demo_patients()
        st.session_state.confirmed_plan = []
        st.session_state.complaints = []
        st.session_state.reminders = []
        st.session_state.integration_events = []
        st.session_state.ops_alerts = []
        st.session_state.notifications = []
        st.sidebar.success("Prototype reset to demo seed data.")
    st.sidebar.download_button(
        label="Download dataset CSV",
        data=st.session_state.patients_df.to_csv(index=False).encode("utf-8"),
        file_name="chemo_patients_export.csv",
        mime="text/csv",
    )

    if menu == "Oncologist Dashboard":
        oncologist_dashboard()
    else:
        timeline = build_patient_timeline(st.session_state.patients_df.copy())
        timeline = apply_safe_rescheduling(timeline)
        ml_scores, ml_forecast = compute_ml_signals(st.session_state.patients_df.copy(), timeline.copy())
        if menu == "Lab Coordinator Dashboard":
            lab_coordinator_dashboard(timeline, ml_scores)
        elif menu == "Daycare Dashboard":
            daycare_dashboard(timeline, ml_scores)
        elif menu == "Pharmacy Dashboard":
            pharmacy_dashboard(timeline, ml_scores)
        elif menu == "Operations & Admin Dashboard":
            operations_dashboard(timeline, ml_forecast)
            st.markdown("---")
            admin_dashboard(timeline, ml_scores)
        elif menu == "Patient Timeline":
            patient_timeline_panel()
        elif menu == "Patient Reminder Panel":
            reminder_panel()
        elif menu == "Patient Companion Interface":
            patient_interface_dashboard()
        elif menu == "Complaint Escalation Panel":
            complaint_panel()


if __name__ == "__main__":
    main()
