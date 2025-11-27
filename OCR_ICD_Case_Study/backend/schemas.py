
from pydantic import BaseModel
from typing import List, Tuple, Optional


class UploadOut(BaseModel):
    status: str
    doc_id: str


class OCRChunk(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    text: str
    # (x1, y1, x2, y2) in source image coordinates
    bbox: Tuple[float, float, float, float]


class ICDItem(BaseModel):
    icd_code: str
    icd_description: str
    supporting_sentence: str


class ExtractRequest(BaseModel):
    doc_id: str


class ExtractResponse(BaseModel):
    doc_id: str
    icds: List[ICDItem]


class SupportingLocation(BaseModel):
    icd_code: str
    icd_description: str
    doc_name: str
    page: int
    bbox: Tuple[float, float, float, float]
    sentence: str


class ReportRequest(BaseModel):
    doc_id: str
    icd_codes: Optional[List[str]] = None  # if None, use all


class ReportResponse(BaseModel):
    doc_id: str
    locations: List[SupportingLocation]
