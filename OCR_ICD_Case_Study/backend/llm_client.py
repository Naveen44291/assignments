
import json
from typing import List

from .schemas import ICDItem
from .config import OPENAI_API_KEY, LLM_MODEL, USE_MOCK_LLM

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def extract_icd_with_llm_mock(doc_text: str) -> List[ICDItem]:
    """Mock ICD extraction matching your test PDF."""
    return [
        ICDItem(
            icd_code="E11.9",
            icd_description="Type 2 diabetes mellitus without complications",
            supporting_sentence="Diagnosis: Type 2 Diabetes Mellitus",
        ),
        ICDItem(
            icd_code="I10",
            icd_description="Essential (primary) hypertension",
            supporting_sentence="Diagnosis: Hypertension",
        ),
    ]


def extract_icd_with_llm(doc_text: str) -> List[ICDItem]:
    """Real OpenAI call (JSON structured output) if configured, else mock.

    Prompt: 'Extract ICD codes and exact supporting sentence.'
    """
    if USE_MOCK_LLM or not OPENAI_API_KEY or OpenAI is None:
        return extract_icd_with_llm_mock(doc_text)

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a medical coding assistant. "
        "Given clinical text, extract ICD-10 codes with their description and the exact supporting sentence. "
        "Return a JSON object with key 'icds' as a list, where each item has: "
        "icd_code, icd_description, supporting_sentence."
    )

    user_prompt = f"Clinical text:\n\n{doc_text}\n\nReturn ONLY JSON."

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    icds_raw = data.get("icds") or []
    icd_items: List[ICDItem] = []
    for item in icds_raw:
        icd_items.append(
            ICDItem(
                icd_code=item.get("icd_code", ""),
                icd_description=item.get("icd_description", item.get("description", "")),
                supporting_sentence=item.get("supporting_sentence", ""),
            )
        )
    return icd_items
