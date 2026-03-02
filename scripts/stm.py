import os
import sys
import json
import re
from pathlib import Path
from contextlib import contextmanager

import streamlit as st
from llama_cpp import Llama

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --------------------------------------------------
# Prompt
# --------------------------------------------------
PROMPT_TEMPLATE = """You are a clinical NLP classifier. Return ONLY valid JSON.

TASK
Determine whether the patient has CURRENT HOUSING INSTABILITY from the clinical note.

DEFINITIONS

housing_instability = "positive" if the note explicitly indicates ANY of:
- Current homelessness / unhoused / no stable place to live or sleep
- Living in shelter, car, street, temporary arrangements due to housing need
- Imminent housing loss (eviction risk, cannot pay rent, staying with others due to housing need)
- Actively seeking or receiving housing assistance or housing case management

housing_instability = "negative" if the note explicitly indicates stable housing, such as:
- Lives at home / apartment / rented or owned residence
- Lives independently or with family/partner without housing concern
- Has stable place to live
- Social needs mentioned WITHOUT housing context (e.g., food, transport only)

housing_instability = "unknown" if:
- Housing status not mentioned
- Too vague or no explicit housing anchor

TEMPORAL RULES
- Classify based ONLY on current status.
- Past or resolved homelessness (e.g., "history of homelessness", "previously homeless") ≠ current instability.
- Past homelessness alone → housing_instability = "positive" with risk_tier = "Low".

SECONDARY TASK (ONLY if housing_instability = "positive")
Assign housing risk tier based on CURRENT severity.

RISK TIER DEFINITIONS (choose highest applicable)

High:
Patient currently homeless / unhoused / no stable place to sleep.
Examples: "currently homeless", "living in shelter", "sleeping in car", "no fixed address", "staying on street"

Medium:
Not currently homeless BUT housing loss imminent OR actively pursuing housing assistance.
Examples: "facing eviction", "risk of losing housing", "cannot pay rent", "applied for housing assistance",
"working with housing case manager", "on housing waitlist", "seeking shelter placement", "staying with friends temporarily"

Low:
Housing instability concern WITHOUT current homelessness or imminent loss.
Includes past homelessness.
Examples: "housing insecure", "unstable housing", "housing concerns", "difficulty maintaining housing",
"history of homelessness", "previously homeless", "formerly homeless now housed", "past shelter use"

OUTPUT RULES
- Use ONLY information explicitly stated in the note.
- Evidence must be verbatim quote(s) from the note.
- risk_tier MUST be one of: "High", "Medium", "Low", or null.
- Assign risk_tier ONLY if housing_instability = "positive".
- If housing_instability ≠ "positive", risk_tier = null.
- Do NOT infer or assume.
- Return JSON only.

JSON SCHEMA
{
    "housing_instability": "positive | negative | unknown",
    "reason_code": "HOUSING_HOMELESS | HOUSING_RISK | HOUSING_ASSISTANCE | NONE | INSUFFICIENT_INFO",
    "risk_tier": "High | Medium | Low | null",
    "evidence": ["verbatim quote"]
}

REASON CODE RULES
- HOUSING_HOMELESS → current homelessness/unhoused
- HOUSING_RISK → eviction risk / unstable / staying with others
- HOUSING_ASSISTANCE → seeking/receiving housing services
- NONE → stable housing
- INSUFFICIENT_INFO → unknown

Clinical note:
<<<
{note}
>>>
"""

# --------------------------------------------------
# Helpers
# --------------------------------------------------
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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON in output:\n{text}")
    return json.loads(text[start:end + 1])

import html
import re

def highlight_evidence(note_text: str, evidence: list[str]) -> str:
    if not note_text:
        return "_(empty)_"

    safe = html.escape(note_text)

    def mark(s: str) -> str:
        return (
            "<mark style='background: rgba(255,235,59,0.6); "
            "padding:2px 4px; border-radius:4px;'>"
            f"{s}</mark>"
        )

    def sub_ci(text, phrase):
        if not phrase.strip():
            return text, 0
        pat = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
        return pat.subn(lambda m: mark(m.group(0)), text)

    # 1) Try exact evidence
    total = 0
    for ev in sorted(evidence, key=len, reverse=True):
        safe, n = sub_ci(safe, ev)
        total += n

    # 2) Fallback keywords if nothing matched
    if total == 0:
        keywords = [
            "homeless",
            "permanent residence",
            "sleeping in his car",
            "staying intermittently",
            "housing resources",
            "transitional housing",
            "shelter",
            "eviction",
        ]

        ev_text = " ".join(evidence).lower()
        for kw in keywords:
            if kw in ev_text:
                safe, _ = sub_ci(safe, kw)

    return safe.replace("\n", "<br/>")




# --------------------------------------------------
# Load LLM (cached, offline)
# --------------------------------------------------
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


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="SDOH – LLM Only (Offline)", layout="wide")
st.title("SDOH Housing Instability – LLM Only (Offline)")

llm = load_llm()

st.markdown("Paste a clinical note below. The model runs **fully offline**.")

note_text = st.text_area(
    "Clinical Note",
    height=260,
    placeholder="Paste clinical text here…"
)

if st.button("Run LLM", type="primary"):
    if not note_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running LLM inference (offline)…"):
            try:
                result = run_llm(llm, note_text)
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.markdown("### Model Output")
        st.json(result)

        evidence = result.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            st.markdown("### Evidence Highlighting")
            highlighted = highlight_evidence(note_text, evidence)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.info("No evidence returned by the model.")
