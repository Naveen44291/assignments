# main.py
import os
import re
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI
from supabase_lib import query_rag, insert_resume

load_dotenv()

# OpenAI client (new SDK style)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not set — endpoints that use OpenAI will fail.")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Defaults (safe, configurable)
DEFAULT_USERNAME = os.environ.get("DEFAULT_USERNAME")
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID")


def extract_metadata_from_text(text: str):
    """
    Extract optional metadata from user's text (conservative regex).
    Supports:
      - book: <document_id>
      - document: <document_id>
      - chapter: <chapter title>
      - para/paragraphs: 5 or 5-10
    Returns dict with document_id, chapter_title, min_paragraph, max_paragraph
    """
    meta = {
        "document_id": None,
        "chapter_title": None,
        "min_paragraph": None,
        "max_paragraph": None
    }

    # document / book
    m = re.search(r"(?:book|document)\s*[:=]\s*([A-Za-z0-9_\- \.]+)", text, flags=re.IGNORECASE)
    if m:
        meta["document_id"] = m.group(1).strip()

    # chapter title or number
    m = re.search(r"(?:chapter|chapter_title)\s*[:=]\s*([A-Za-z0-9_\- \.]+)", text, flags=re.IGNORECASE)
    if m:
        meta["chapter_title"] = m.group(1).strip()

    # paragraph range like "paragraphs: 5-10" or "para: 7"
    m = re.search(r"(?:paragraphs|paragraph|para)\s*[:=]\s*([0-9]+)\s*-\s*([0-9]+)", text, flags=re.IGNORECASE)
    if m:
        meta["min_paragraph"] = int(m.group(1))
        meta["max_paragraph"] = int(m.group(2))
    else:
        m = re.search(r"(?:paragraphs|paragraph|para)\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)
        if m:
            meta["min_paragraph"] = int(m.group(1))
            meta["max_paragraph"] = int(m.group(1))

    return meta


def get_embedding(text: str):
    if not client:
        raise RuntimeError("OpenAI client is not configured (set OPENAI_API_KEY).")
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def compose_answer_with_contexts(user_query: str, contexts: list):
    """
    Uses LLM (if available) to generate a final answer from contexts.
    Falls back to concatenation if client not available.
    """
    if not client:
        return "OpenAI not configured. Retrieved contexts:\n\n" + "\n\n---\n\n".join(contexts)
    system_msg = "You are a helpful assistant. Use the provided context snippets from books to answer the user's question concisely."
    full_context = "\n\n".join(contexts) if contexts else "No context available."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n\n{full_context}\n\nUser question: {user_query}"}
    ]
    completion = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0)
    return completion.choices[0].message.content


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return HTMLResponse("<h3>RAG App</h3>")


@app.post("/api/chat")
async def chat_endpoint(
    request: Request,
    message: str = Form(...),
    # Explicit metadata fields (exposed in UI or API)
    document_types: str = Form(None),   # comma separated, e.g. "book"
    document_id: str = Form(None),
    chapter_title: str = Form(None),
    min_paragraph: int = Form(None),
    max_paragraph: int = Form(None),
    username: str = Form(None),
    user_id: str = Form(None),
    top_k: int = Form(8),
):
    """
    Accepts form-encoded fields (also works with JSON if you call body parsing).
    - message: user's query (can include inline metadata like 'book:...' if you prefer)
    - document_types: optional comma-separated types (default: "book")
    - document_id/chapter_title/min_paragraph/max_paragraph: optional filters
    - username/user_id: optional tenant filters (strongly recommended in multi-tenant setups)
    - top_k: how many matches to retrieve
    """

    # prefer explicit form inputs; if not provided, try to extract from message
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    extracted = extract_metadata_from_text(message)
    # use explicit params if provided, otherwise fall back to extraction
    document_id = document_id or extracted.get("document_id")
    chapter_title = chapter_title or extracted.get("chapter_title")
    min_paragraph = min_paragraph if min_paragraph is not None else extracted.get("min_paragraph")
    max_paragraph = max_paragraph if max_paragraph is not None else extracted.get("max_paragraph")

    # parse document_types: string -> list
    if document_types:
        doc_types = [d.strip() for d in document_types.split(",") if d.strip()]
    else:
        doc_types = ["book"]

    # prefer explicit username/user_id → fall back to environment defaults → else None
    username = username or os.environ.get("DEFAULT_USERNAME") or DEFAULT_USERNAME
    user_id = user_id or os.environ.get("DEFAULT_USER_ID") or DEFAULT_USER_ID

    try:
        query_embedding = get_embedding(message)
    except Exception as e:
        return JSONResponse({"error": f"Failed to create embedding: {str(e)}"}, status_code=500)

    # Call Supabase RPC with all metadata filters supported by SQL
    resp = query_rag(
        query_embedding=query_embedding,
        match_count=top_k,
        document_types=doc_types,
        chapter_title=chapter_title,
        document_id=document_id,
        min_paragraph=min_paragraph,
        max_paragraph=max_paragraph,
        username=username,
        user_id=user_id
    )

    if resp.error:
        return JSONResponse({"error": f"Database error: {resp.error}"}, status_code=500)

    rows = resp.data or []
    contexts = [r.get("context", "") for r in rows]

    answer = compose_answer_with_contexts(message, contexts)

    return {
        "query": message,
        "metadata_filters": {
            "document_types": doc_types,
            "document_id": document_id,
            "chapter_title": chapter_title,
            "min_paragraph": min_paragraph,
            "max_paragraph": max_paragraph,
            "username": username,
            "user_id": user_id,
            "top_k": top_k
        },
        "rag_hits": rows,
        "answer": answer
    }


# Simple resume parser kept (unchanged behavior)
@app.post("/api/parse-resume")
async def parse_resume(request: Request):
    body = await request.json()
    html_content = body.get("html_content", "")
    if not html_content:
        return JSONResponse({"error": "No html_content provided"}, status_code=400)

    if client:
        system_prompt = "You are a resume parser. Extract name, contact, summary, work_experience, education, skills into JSON."
        user_prompt = f"Parse HTML and return only JSON:\n\n{html_content}"
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0
        )
        parsed = completion.choices[0].message.content
        try:
            parsed_json = json.loads(parsed)
        except Exception:
            parsed_json = {"parsed_text": parsed}
        # optionally insert into DB (commented out to be non-destructive)
        # insert_resume(parsed_json)
        return {"parsed_resume": parsed_json}
    else:
        return {"parsed_resume": {"raw_html": html_content}}
