
import time
from typing import List, Tuple

import requests

from .schemas import OCRChunk
from .config import AZURE_OCR_ENDPOINT, AZURE_OCR_KEY, USE_MOCK_OCR


def _polygon_to_bbox(poly):
    """Convert Azure 8-point polygon [x1,y1,...,x4,y4] to (x1,y1,x2,y2)."""
    xs = poly[0::2]
    ys = poly[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def run_azure_ocr_mock(doc_id: str, doc_name: str) -> Tuple[List[OCRChunk], int, int]:
    """Mock OCR with fixed text + bounding boxes.

    Page size for mock is 800x1000. Bounding boxes are in that coordinate space.
    """
    page_width, page_height = 800, 1000
    chunks = [
        OCRChunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page=1,
            text="The patient has Type 2 Diabetes Mellitus without complications.",
            bbox=(80, 200, 720, 240),
        ),
        OCRChunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page=1,
            text="Hypertension is also noted with controlled blood pressure.",
            bbox=(80, 260, 720, 300),
        ),
        OCRChunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page=1,
            text="Follow-up visit recommended in 3 months.",
            bbox=(80, 320, 720, 360),
        ),
    ]
    return chunks, page_width, page_height


def run_azure_ocr_bytes(doc_id: str, doc_name: str, content: bytes) -> Tuple[List[OCRChunk], int, int]:
    """Real Azure OCR call using Vision Read API (v3.2).

    Requires:
      - AZURE_OCR_ENDPOINT like https://YOUR_RESOURCE.cognitiveservices.azure.com
      - AZURE_OCR_KEY

    Uses /vision/v3.2/read/analyze (async).
    """
    if not AZURE_OCR_ENDPOINT or not AZURE_OCR_KEY:
        raise RuntimeError("Azure OCR endpoint/key not configured. Set AZURE_OCR_ENDPOINT and AZURE_OCR_KEY.")

    analyze_url = f"{AZURE_OCR_ENDPOINT}/vision/v3.2/read/analyze"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_OCR_KEY,
        "Content-Type": "application/octet-stream",
    }

    # Submit image bytes
    response = requests.post(analyze_url, headers=headers, data=content)
    response.raise_for_status()

    # Get operation URL from header
    operation_url = response.headers.get("Operation-Location")
    if not operation_url:
        raise RuntimeError("Missing Operation-Location header from Azure OCR response.")

    # Poll for result
    result = None
    for _ in range(20):
        time.sleep(1.0)
        r = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY})
        r.raise_for_status()
        result = r.json()
        if result.get("status") == "succeeded":
            break
    else:
        raise RuntimeError("Azure OCR did not finish in time.")

    analyze_result = result.get("analyzeResult") or {}
    read_results = analyze_result.get("readResults") or []

    chunks: List[OCRChunk] = []
    page_width = 800
    page_height = 1000

    for page in read_results:
        page_number = page.get("page", 1)
        page_width = page.get("width", page_width)
        page_height = page.get("height", page_height)
        lines = page.get("lines") or []
        for line in lines:
            text = line.get("text", "")
            bb = line.get("boundingBox") or []
            if len(bb) >= 8:
                bbox = _polygon_to_bbox(bb)
            else:
                # Fallback if structure changes
                bbox = (0.0, 0.0, float(page_width), float(page_height))
            chunks.append(
                OCRChunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_number,
                    text=text,
                    bbox=bbox,
                )
            )

    return chunks, int(page_width), int(page_height)


def run_ocr(doc_id: str, doc_name: str, content: bytes) -> Tuple[List[OCRChunk], int, int]:
    """Decide between mock OCR and real Azure OCR based on USE_MOCK_OCR flag."""
    if USE_MOCK_OCR:
        return run_azure_ocr_mock(doc_id, doc_name)
    return run_azure_ocr_bytes(doc_id, doc_name, content)
