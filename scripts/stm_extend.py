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

# =========================
# SCHEMA (YOUR REAL COLUMNS)
# =========================
PATIENT_COL = "PATIENTDURABLEKEY"
TEXT_COL    = "NOTETEXT"
DATE_COL    = "creation_date"
TYPE_COL    = "NOTETYPE"
AUTHOR_COL  = "NOTEAUTHORTYPE"

# Optional (if present in your df, we will use it)
NOTE_ID_CANDIDATES = ["CLINICALNOTEKEY", "NOTEKEY", "NOTE_ID", "ClinicalNoteKey"]

# =========================
# PATHS
# =========================
BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Point this to your notes file
# If you're not using a CSV, replace this load step with your own df.
DATA_PATH  = BASE_DIR / "data" / "notes.csv"

CACHE_DIR  = BASE_DIR / "data" / "patient_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PROMPT
# =========================
PROMPT_TEMPLATE = """You are a clinical NLP classifier. Return ONLY valid JSON.

Task: Determine whether the patient has HOUSING INSTABILITY.

Definition (positive):
- Homelessness, unhoused, shelter use
- Lack of stable housing / unstable living situation
- Risk of losing housing / eviction
- Staying with others due to housing need
- Housing assistance/resources requested or provided
- Social work/case management for housing/homelessness evaluation

Negative (stable housing):
- Explicit statements indicating stable housing, such as:
  - lives at home
  - lives alone / by himself / by herself (as stable living, not couch surfing)
  - lives with spouse/partner/family
  - owns or rents a home/apartment
  - has a stable place to live
- No housing-related need mentioned

Unknown:
- Housing status not mentioned OR too vague OR no explicit housing anchor

Temporal rule:
- Past homelessness alone (e.g., "previously homeless", "history of homelessness") does NOT imply CURRENT housing instability.
- Classify based on CURRENT status.

Rules:
- Use ONLY information explicitly stated in the note.
- Evidence MUST be a verbatim quote from the note.
- Output JSON only (no markdown, no extra text).

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

# =========================
# HELPERS
# =========================
@contextmanager
def suppress_stderr():
    old = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = old


def extract_json(text: str) -> dict:
    """Extract first {...} block and json.loads it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON in output:\n{text}")
    return json.loads(text[start:end + 1])


def highlight_evidence(note_text: str, evidence: list[str]) -> str:
    """
    Highlights evidence spans inside the note text.
    Works best when evidence strings are verbatim substrings.
    """
    if not note_text:
        return "_(empty)_"

    safe = html.escape(note_text)

    def mark(s: str) -> str:
        return (
            "<mark style='background:rgba(255,235,59,0.6);"
            "padding:2px 4px;border-radius:4px;'>"
            f"{s}</mark>"
        )

    total_matches = 0
    for ev in sorted(evidence or [], key=len, reverse=True):
        if not ev or not ev.strip():
            continue
        ev_esc = html.escape(ev)
        pat = re.compile(re.escape(ev_esc), re.IGNORECASE)
        safe, n = pat.subn(lambda m: mark(m.group(0)), safe)
        total_matches += n

    return safe.replace("\n", "<br/>")


def get_note_id_column(df: pd.DataFrame) -> str | None:
    for c in NOTE_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


# =========================
# LLM LOADING (OFFLINE)
# =========================
@st.cache_resource
def load_llm():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at:\n{MODEL_PATH}")
        st.stop()

    with suppress_stderr():
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,   # Metal on Apple Silicon
            verbose=False,
        )
    return llm


def run_llm(llm: Llama, note: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(note=note)

    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    text = out["choices"][0]["text"].strip()
    return extract_json(text)


# =========================
# OPTION 2: DISK CACHE
# =========================
def cache_path(patient_key: str) -> Path:
    # safe filename
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(patient_key))
    return CACHE_DIR / f"{safe}.json"


def save_patient_cache(patient_key: str, results_df: pd.DataFrame, summary: dict):
    payload = {
        "patient_key": str(patient_key),
        "created_at_utc": datetime.utcnow().isoformat(),
        "summary": summary,
        "results": results_df.to_dict(orient="records"),
    }
    with open(cache_path(patient_key), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_patient_cache(patient_key: str):
    path = cache_path(patient_key)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def clear_patient_cache(patient_key: str):
    path = cache_path(patient_key)
    if path.exists():
        path.unlink()


# =========================
# PATIENT PIPELINE
# =========================
def process_patient(df: pd.DataFrame, patient_key: str, llm: Llama) -> tuple[pd.DataFrame, dict]:
    # Filter patient
    rows = df[df[PATIENT_COL].astype(str) == str(patient_key)].copy()
    if rows.empty:
        raise ValueError(f"No notes found for {PATIENT_COL} = {patient_key}")

    # Note id column (optional)
    note_id_col = get_note_id_column(rows)

    # Ensure date parse
    rows[DATE_COL] = pd.to_datetime(rows[DATE_COL], errors="coerce")

    # Run LLM per note
    out_rows = []
    for idx, r in rows.iterrows():
        note_text = str(r.get(TEXT_COL, "") or "")
        if not note_text.strip():
            # Skip/mark empty notes
            llm_res = {
                "housing_instability": "unknown",
                "reason_code": "INSUFFICIENT_INFO",
                "evidence": [],
            }
        else:
            llm_res = run_llm(llm, note_text)

        out_rows.append({
            "patient_key": str(patient_key),
            "note_id": str(r.get(note_id_col, idx)) if note_id_col else str(idx),
            "creation_date": r.get(DATE_COL, None),
            "NOTETYPE": r.get(TYPE_COL, None),
            "NOTEAUTHORTYPE": r.get(AUTHOR_COL, None),
            "NOTETEXT": note_text,
            **llm_res
        })

    res_df = pd.DataFrame(out_rows)
    res_df["creation_date"] = pd.to_datetime(res_df["creation_date"], errors="coerce")
    res_df = res_df.sort_values("creation_date", ascending=True)

    positives = res_df[res_df["housing_instability"] == "positive"].copy()
    negatives = res_df[res_df["housing_instability"] == "negative"].copy()
    unknowns  = res_df[res_df["housing_instability"] == "unknown"].copy()

    most_recent = res_df.iloc[-1] if len(res_df) else None
    most_recent_positive = positives.iloc[-1] if len(positives) else None

    summary = {
        "patientdurablekey": str(patient_key),
        "total_notes": int(len(res_df)),
        "positive_notes": int(len(positives)),
        "negative_notes": int(len(negatives)),
        "unknown_notes": int(len(unknowns)),
        "positive_dates": positives["creation_date"].dt.date.astype(str).tolist() if len(positives) else [],
        "most_recent_status": str(most_recent["housing_instability"]) if most_recent is not None else None,
        "most_recent_date": str(most_recent["creation_date"].date()) if most_recent is not None and pd.notnull(most_recent["creation_date"]) else None,
        "most_recent_reason_code": str(most_recent["reason_code"]) if most_recent is not None else None,
        "most_recent_evidence": most_recent["evidence"] if most_recent is not None else [],
        "most_recent_positive_date": (
            str(most_recent_positive["creation_date"].date())
            if most_recent_positive is not None and pd.notnull(most_recent_positive["creation_date"])
            else None
        ),
        "most_recent_positive_evidence": most_recent_positive["evidence"] if most_recent_positive is not None else [],
    }

    return res_df, summary


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="SDOH – Patient Analysis (Offline)", layout="wide")
st.title("SDOH – Patient Analysis (Offline)")

# Load data
if not DATA_PATH.exists():
    st.error(f"Data file not found at:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Validate schema EARLY (no silent failures)
required = {PATIENT_COL, TEXT_COL, DATE_COL, TYPE_COL, AUTHOR_COL}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Available columns:", list(df.columns))
    st.stop()

llm = load_llm()

# Sidebar controls
st.sidebar.header("Patient Lookup")
patient_key = st.sidebar.text_input("Enter PATIENTDURABLEKEY")

colA, colB = st.sidebar.columns(2)
run_btn = colA.button("Run Patient Analysis")
force_btn = colB.button("Force Re-run")

if run_btn or force_btn:
    if not patient_key.strip():
        st.warning("Please enter a PATIENTDURABLEKEY.")
        st.stop()

    if force_btn:
        clear_patient_cache(patient_key)

    cached = load_patient_cache(patient_key)

    if cached is not None:
        st.info("Loaded cached results from disk.")
        res_df = pd.DataFrame(cached["results"])
        summary = cached["summary"]
        # Ensure types
        res_df["creation_date"] = pd.to_datetime(res_df["creation_date"], errors="coerce")
    else:
        with st.spinner("Running LLM across all notes (offline)…"):
            res_df, summary = process_patient(df, patient_key, llm)
            save_patient_cache(patient_key, res_df, summary)

    # -------------------------
    # Summary
    # -------------------------
    st.subheader("Patient Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Notes", summary["total_notes"])
    m2.metric("Positive Notes", summary["positive_notes"])
    m3.metric("Negative Notes", summary["negative_notes"])
    m4.metric("Unknown Notes", summary["unknown_notes"])

    st.json(summary)

    # -------------------------
    # Timeline table (quick)
    # -------------------------
    st.subheader("Timeline")
    show_cols = ["creation_date", "housing_instability", "reason_code", "note_id", "NOTETYPE", "NOTEAUTHORTYPE"]
    show_cols = [c for c in show_cols if c in res_df.columns]
    st.dataframe(res_df.sort_values("creation_date", ascending=False)[show_cols], use_container_width=True)

    # -------------------------
    # Note-level detail with highlighting
    # -------------------------
    st.subheader("Note-level Results (with Evidence Highlighting)")

    for _, r in res_df.sort_values("creation_date", ascending=False).iterrows():
        date_str = ""
        if pd.notnull(r.get("creation_date")):
            date_str = str(pd.to_datetime(r["creation_date"]).date())

        status = str(r.get("housing_instability", "")).upper()
        reason = str(r.get("reason_code", ""))

        header = f"{date_str} — {status} — {reason} — NoteID: {r.get('note_id')}"
        with st.expander(header):
            st.json({
                "housing_instability": r.get("housing_instability"),
                "reason_code": r.get("reason_code"),
                "evidence": r.get("evidence", []),
                "NOTETYPE": r.get("NOTETYPE"),
                "NOTEAUTHORTYPE": r.get("NOTEAUTHORTYPE"),
                "creation_date": date_str,
                "note_id": r.get("note_id"),
            })

            evidence = r.get("evidence", [])
            note_text = r.get("NOTETEXT", "")

            if isinstance(evidence, list) and len(evidence) > 0:
                st.markdown(highlight_evidence(note_text, evidence), unsafe_allow_html=True)
            else:
                st.markdown(html.escape(note_text).replace("\n", "<br/>"), unsafe_allow_html=True)
                st.info("No evidence returned by the model.")
