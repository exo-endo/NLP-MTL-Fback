import os
import sys
import json
import re
import html
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import streamlit as st
from llama_cpp import Llama

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DATA_PATH = BASE_DIR / "data" / "notes.csv"
CACHE_DIR = BASE_DIR / "data" / "patient_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# --------------------------------------------------
# Prompt
# --------------------------------------------------
PROMPT_TEMPLATE = """You are a clinical NLP classifier. Return ONLY valid JSON.

Task: Determine whether the patient has HOUSING INSTABILITY.

Definition (positive):
- Homelessness, unhoused, shelter use
- Lack of stable housing / unstable living situation
- Risk of losing housing / eviction
- Staying with others due to housing need
- Housing assistance/resources requested or provided

Negative (stable housing):
- Explicit statements of stable housing (lives at home, lives with spouse, owns/rents home)
- No housing-related need mentioned

Unknown:
- Too vague to tell OR no explicit housing anchor

Temporal rule:
- Past homelessness alone does NOT imply current housing instability.
- Classify based on CURRENT status only.

Rules:
- Use ONLY information explicitly stated.
- Evidence MUST be a verbatim quote.
- Output JSON only.

JSON schema:
{{
"housing_instability": "positive|negative|unknown",
"reason_code": "HOUSING_ASSISTANCE_REQUEST|HOMELESSNESS_MENTIONED|HOUSING_RISK|NONE|INSUFFICIENT_INFO",
"evidence": ["verbatim quote"]
}}

Clinical note:
<<<
{note}
>>>
"""

# --------------------------------------------------
# Utilities
# --------------------------------------------------
@contextmanager
def suppress_stderr():
    old = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found:\n{text}")
    return json.loads(text[start:end + 1])


def highlight_evidence(note_text: str, evidence: list[str]) -> str:
    safe = html.escape(note_text)

    def mark(s):
        return (
            "<mark style='background:rgba(255,235,59,0.6);"
            "padding:2px 4px;border-radius:4px;'>"
            f"{s}</mark>"
        )

    for ev in sorted(evidence, key=len, reverse=True):
        if not ev.strip():
            continue
        pat = re.compile(re.escape(html.escape(ev)), re.IGNORECASE)
        safe = pat.sub(lambda m: mark(m.group(0)), safe)

    return safe.replace("\n", "<br/>")


# --------------------------------------------------
# LLM
# --------------------------------------------------
@st.cache_resource
def load_llm():
    with suppress_stderr():
        return Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False,
        )


def run_llm(llm: Llama, note: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(note=note)
    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    return extract_json(out["choices"][0]["text"].strip())


# --------------------------------------------------
# Caching (Option 2)
# --------------------------------------------------
def cache_path(patient_id: str) -> Path:
    return CACHE_DIR / f"{patient_id}.json"


def save_patient(patient_id: str, notes_df: pd.DataFrame, summary: dict):
    payload = {
        "patient_id": patient_id,
        "summary": summary,
        "notes": notes_df.to_dict(orient="records"),
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(cache_path(patient_id), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_patient(patient_id: str):
    path = cache_path(patient_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------
# Patient Processing
# --------------------------------------------------
def process_patient(df: pd.DataFrame, patient_id: str, llm: Llama):
    rows = df[df["patient_id"] == patient_id].copy()
    results = []

    for _, r in rows.iterrows():
        res = run_llm(llm, r["note_text"])
        results.append({
            **r,
            **res
        })

    res_df = pd.DataFrame(results)
    res_df["note_date"] = pd.to_datetime(res_df["note_date"])

    positives = res_df[res_df["housing_instability"] == "positive"]

    summary = {
        "total_notes": len(res_df),
        "positive_notes": len(positives),
        "positive_dates": positives["note_date"].dt.date.astype(str).tolist(),
        "most_recent_positive_note": (
            positives.sort_values("note_date").iloc[-1]["note_text"]
            if not positives.empty else None
        )
    }

    return res_df, summary


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config("SDOH Patient View (Offline)", layout="wide")
st.title("SDOH – Patient-Level Housing Instability (Offline)")

df = pd.read_csv(DATA_PATH)
llm = load_llm()

patient_id = st.text_input("Enter Patient ID")

if st.button("Run Patient Analysis", type="primary"):
    if not patient_id:
        st.warning("Enter a patient ID.")
        st.stop()

    cached = load_patient(patient_id)

    if cached:
        st.info("Loaded cached results.")
        res_df = pd.DataFrame(cached["notes"])
        summary = cached["summary"]
    else:
        with st.spinner("Running LLM across patient notes…"):
            res_df, summary = process_patient(df, patient_id, llm)
            save_patient(patient_id, res_df, summary)

    # -------------------------
    # Summary
    # -------------------------
    st.subheader("Patient Summary")
    st.json(summary)

    # -------------------------
    # Notes
    # -------------------------
    st.subheader("Note-level Results")

    for _, r in res_df.sort_values("note_date", ascending=False).iterrows():
        with st.expander(f"{r['note_date'].date()} — {r['housing_instability'].upper()}"):
            st.json({
                "housing_instability": r["housing_instability"],
                "reason_code": r["reason_code"],
                "evidence": r["evidence"],
            })
            st.markdown(
                highlight_evidence(r["note_text"], r["evidence"]),
                unsafe_allow_html=True
            )
