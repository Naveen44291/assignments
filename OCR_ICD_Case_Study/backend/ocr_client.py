import time
from typing import List, Tuple
import requests
from .schemas import OCRChunk
from .config import AZURE_OCR_ENDPOINT, AZURE_OCR_KEY, USE_MOCK_OCR


# def run_azure_ocr_mock(doc_id: str, doc_name: str):
#     page_width, page_height = 800, 1000
#     chunks = [
#         OCRChunk(doc_id=doc_id, doc_name=doc_name, page=1,
#                  text="The patient has Type 2 Diabetes Mellitus without complications.",
#                  bbox=(80, 200, 720, 240)),
#         OCRChunk(doc_id=doc_id, doc_name=doc_name, page=1,
#                  text="Hypertension is also noted with controlled blood pressure.",
#                  bbox=(80, 260, 720, 300)),
#         OCRChunk(doc_id=doc_id, doc_name=doc_name, page=1,
#                  text="Follow-up visit recommended in 3 months.",
#                  bbox=(80, 320, 720, 360)),
#     ]
#     return chunks, page_width, page_height


def run_azure_ocr_bytes(doc_id: str, doc_name: str, content: bytes):
    """
    Azure Document Intelligence Read API (v4.0)
    Supports: PDF, JPG, PNG, TIFF
    """

    if not AZURE_OCR_ENDPOINT or not AZURE_OCR_KEY:
        raise RuntimeError("Missing Azure config")

    endpoint = AZURE_OCR_ENDPOINT.rstrip("/")
    url = f"{endpoint}/formrecognizer/documentModels/prebuilt-read:analyze?api-version=2023-07-31"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_OCR_KEY,
        "Content-Type": "application/pdf"  # works for image or pdf
    }

    print("\n📤 Sending file to Azure PDF OCR...")

    response = requests.post(url, headers=headers, data=content)

    if response.status_code != 202:
        raise RuntimeError(f"Azure OCR submit failed ({response.status_code}): {response.text}")

    operation_location = response.headers["Operation-Location"]

    print("⏳ Waiting for OCR result...")

    # Poll for result
    for _ in range(30):
        result_resp = requests.get(operation_location, headers={"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY})
        if result_resp.status_code == 200:
            result_json = result_resp.json()
            status = result_json.get("status")
            if status == "succeeded":
                break
            if status == "failed":
                raise RuntimeError("Azure OCR processing failed.")
        time.sleep(1)

    print("✅ Azure OCR completed!\n")

    result = result_json["analyzeResult"]
    pages = result.get("pages", [])

    chunks = []

    print("========================")
    print("🔍 OCR BOUNDING BOX DEBUG")
    print("========================")

    for page in pages:
        page_number = page.get("pageNumber", 1)
        width = page.get("width")
        height = page.get("height")

        print(f"\n--- PAGE {page_number} ---")
        print(f"Page Size = {width} x {height}")

        for line in page.get("lines", []):
            text = line.get("content", "")
            polygon = line.get("polygon", [])

            if polygon:
                xs = [polygon[i] for i in range(0, len(polygon), 2)]
                ys = [polygon[i] for i in range(1, len(polygon), 2)]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (0, 0, width, height)

            print(f"TEXT: {text}")
            print(f"POLYGON: {polygon}")
            print(f"BOUNDING BOX: {bbox}\n")

            chunks.append(
                OCRChunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_number,
                    text=text,
                    bbox=bbox,
                )
            )

    print("========================")
    print("END OF BOUNDING BOX DEBUG")
    print("========================\n")

    return chunks, width, height


def run_ocr(doc_id: str, doc_name: str, content: bytes):
    # if USE_MOCK_OCR:
    #     return run_azure_ocr_mock(doc_id, doc_name)

    return run_azure_ocr_bytes(doc_id, doc_name, content)
