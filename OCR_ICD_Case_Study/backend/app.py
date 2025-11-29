import base64
import uuid
from io import BytesIO
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    UploadOut,
    ExtractRequest,
    ExtractResponse,
    ReportRequest,
    ReportResponse,
)
from .ocr_client import run_ocr
from .llm_client import extract_icd_with_llm
from .report_generator import generate_report_for_icds

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


app = FastAPI(title="Pharma ICD OCR Demo (LangChain + Azure + OpenAI + pgvector)")

# Allow simple static frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory doc store for demo
DOC_STORE: Dict[str, Dict[str, Any]] = {}


@app.post("/upload", response_model=UploadOut)
async def upload(file: UploadFile = File(...)):
    """
    Upload an image or PDF → run OCR → return preview image + chunks + bounding boxes.
    """
    content = await file.read()
    doc_id = str(uuid.uuid4())
    doc_name = file.filename or "document"

    # OCR
    chunks, page_width, page_height = run_ocr(doc_id, doc_name, content)

    # PREVIEW image generation
    image_data_url = _make_preview_image_data_url(content, page_width, page_height)

    DOC_STORE[doc_id] = {
        "doc_name": doc_name,
        "chunks": chunks,
        "image_data_url": image_data_url,
        "page_width": page_width,
        "page_height": page_height,
    }

    return UploadOut(status="ok", doc_id=doc_id)


# --------------------------------------------------------------------
# FIXED PREVIEW IMAGE FUNCTION (handles PDF or Images)
# --------------------------------------------------------------------
from PIL import Image as PILImage

def _make_preview_image_data_url(content: bytes, page_width: float, page_height: float) -> str:
    """
    Creates a preview image:
    - If it's an image: load directly
    - If it's a PDF: create a blank white canvas sized based on page inches
    """

    try:
        # Try treating the input as an image
        img = PILImage.open(BytesIO(content)).convert("RGB")

    except Exception:
        # PDF fallback
        print("⚠️ Uploaded file is not an image. Creating blank preview canvas instead.")

        # Convert PDF size from inches → pixels
        # Azure often returns page dimensions in inches
        w_px = int(float(page_width) * 96)
        h_px = int(float(page_height) * 96)

        # Minimum for preview
        w_px = max(w_px, 800)
        h_px = max(h_px, 1100)

        img = PILImage.new("RGB", (w_px, h_px), color="white")

    # Convert to Base64
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.post("/extract-icd", response_model=ExtractResponse)
async def extract_icd(req: ExtractRequest):
    """Run LLM (or mock) to extract ICD codes and supporting sentences."""
    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise ValueError("Unknown doc_id")

    chunks = doc["chunks"]
    full_text = "\n".join(c.text for c in chunks)

    icd_items = extract_icd_with_llm(full_text)
    return ExtractResponse(doc_id=req.doc_id, icds=icd_items)


@app.post("/view-report", response_model=ReportResponse)
async def view_report(req: ReportRequest):
    """Return ICD codes + grounded locations (page, bbox, doc)."""
    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise ValueError("Unknown doc_id")

    chunks = doc["chunks"]
    full_text = "\n".join(c.text for c in chunks)
    icd_items = extract_icd_with_llm(full_text)

    locations = generate_report_for_icds(
        doc_id=req.doc_id,
        chunks=chunks,
        icds=icd_items,
        filter_codes=req.icd_codes,
    )
    return ReportResponse(doc_id=req.doc_id, locations=locations)


@app.get("/doc/{doc_id}")
async def get_doc(doc_id: str):
    """Return preview image + OCR chunks + page dimensions for drawing boxes."""
    doc = DOC_STORE.get(doc_id)
    if not doc:
        return {"error": "doc not found"}

    chunks = [c.model_dump() for c in doc["chunks"]]
    return {
        "doc_id": doc_id,
        "doc_name": doc["doc_name"],
        "image_data_url": doc["image_data_url"],
        "page_width": doc["page_width"],
        "page_height": doc["page_height"],
        "chunks": chunks,
    }
