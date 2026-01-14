# scripts/run_sdoh_llamacpp.py
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

PROMPT_TEMPLATE = """You are a clinical NLP classifier. Return ONLY valid JSON.

Task: Determine whether the patient has HOUSING INSTABILITY.

Definition (positive):
- Homelessness, unhoused, shelter use
- Lack of stable housing / unstable living situation
- Risk of losing housing / eviction / staying with others due to housing need
- Housing assistance/resources requested or provided
- Social work/case management for housing or homelessness evaluation

Negative:
- No housing-related need mentioned
- Social work/resources for non-housing needs only (transport, food, insurance) WITHOUT housing context

Unknown:
- Too vague to tell OR no explicit housing anchor in the note

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


@contextmanager
def suppress_stderr():
    """Suppress stderr noise (Metal/kernel init logs) during model load."""
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = old_stderr


def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from generated text.
    This is robust to occasional extra tokens before/after JSON.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON in output:\n{text}")
    return json.loads(text[start : end + 1])


def build_llm() -> Llama:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    # Suppress Metal/kernel init chatter on Apple Silicon
    with suppress_stderr():
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,        # increase to 8192 only if you truly need longer context
            n_threads=8,       # tune based on your CPU cores; 6-8 is usually fine on M1
            n_gpu_layers=-1,   # use Metal acceleration when available
            verbose=False,     # reduce llama-cpp logging
        )
    return llm


def classify_note(llm: Llama, note: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(note=note)

    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\n>"],
    )

    text = out["choices"][0]["text"].strip()
    return extract_json(text)


if __name__ == "__main__":
    llm = build_llm()

    # Example note (replace with your own, or batch-load later)
    note = (
        "Heather requesting a return call regarding Housing Assistance "
        "that she was working on with someone in the office."
    )

    parsed = classify_note(llm, note)

    # Print ONLY the parsed JSON (clean terminal output)
    print(json.dumps(parsed, indent=2))
