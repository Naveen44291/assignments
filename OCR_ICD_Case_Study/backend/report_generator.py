from typing import List, Optional

from .schemas import ICDItem, SupportingLocation, OCRChunk
from .rag_retriever import build_retriever


def generate_report_for_icds(
    doc_id: str,
    chunks: List[OCRChunk],
    icds: List[ICDItem],
    filter_codes: Optional[List[str]] = None,
) -> List[SupportingLocation]:

    retriever = build_retriever(chunks)
    locations: List[SupportingLocation] = []

    for item in icds:

        # Filter by ICD code if UI passed filters
        if filter_codes and item.icd_code not in filter_codes:
            continue

        # Skip if no supporting sentence
        if not item.supporting_sentence:
            continue

        # Query mock retriever
        docs = retriever.invoke(item.supporting_sentence)
        if not docs:
            continue

        d = docs[0]  # best match (simple logic)

        meta = d["metadata"]            # ← FIXED
        sentence = d["page_content"]    # ← FIXED

        locations.append(
            SupportingLocation(
                icd_code=item.icd_code,
                icd_description=item.icd_description,
                doc_name=meta.get("doc_name", ""),
                page=int(meta.get("page", 1)),
                bbox=tuple(meta.get("bbox", (0, 0, 0, 0))),
                sentence=sentence,
            )
        )

    return locations
