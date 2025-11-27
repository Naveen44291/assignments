import os
from dotenv import load_dotenv, find_dotenv

# --- Load .env properly ---
load_dotenv(find_dotenv(), override=True)

# --- Azure OCR ---
AZURE_OCR_ENDPOINT = os.getenv("AZURE_OCR_ENDPOINT", "").rstrip("/")
AZURE_OCR_KEY = os.getenv("AZURE_OCR_KEY", "")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Supabase / PGVector ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "")

# --- Feature flags ---
USE_MOCK_OCR = os.getenv("USE_MOCK_OCR", "true").lower() == "true"
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
USE_PG_VECTOR = os.getenv("USE_PG_VECTOR", "false").lower() == "true"

# --- Debug printout ---
print(f"[CONFIG] Loaded .env from: {find_dotenv()}")
print(f"[CONFIG] AZURE_OCR_ENDPOINT = {AZURE_OCR_ENDPOINT}")
print(f"[CONFIG] USE_MOCK_OCR = {USE_MOCK_OCR}")
print(f"[CONFIG] USE_MOCK_LLM = {USE_MOCK_LLM}")
print(f"[CONFIG] USE_PG_VECTOR = {USE_PG_VECTOR}")
