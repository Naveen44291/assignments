# supabase_lib.py
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment")
supabase: Client = create_client(supabase_url, supabase_key)


def query_rag(
    query_embedding,
    match_count=5,
    document_types=None,
    chapter_title=None,
    document_id=None,
    min_paragraph=None,
    max_paragraph=None,
    username=None,
    user_id=None,
):
    """
    Call match_rag RPC in Supabase with all supported metadata filters.
    All filter_* arguments default to None and are passed-through.
    """
    payload = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "query_document_types": document_types,
        "filter_username": username,
        "filter_user_id": user_id,
        "filter_document_id": document_id,
        "filter_chapter_title": chapter_title,
        "min_paragraph": min_paragraph,
        "max_paragraph": max_paragraph
    }

    resp = supabase.rpc("match_rag", payload).execute()
    return resp


def insert_resume(resume_json: dict) -> dict:
    """
    Insert parsed resume JSON into the 'resumes' table.
    """
    if not isinstance(resume_json, dict):
        raise ValueError("resume_json must be a dict")
    response = supabase.table("resumes").insert({"resume": resume_json}).execute()
    if response.error:
        raise RuntimeError(f"Failed to insert resume: {response.error}")
    return response.data[0]
