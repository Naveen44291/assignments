
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import FakeEmbeddings

from .schemas import OCRChunk
from .config import OPENAI_API_KEY, SUPABASE_URL, USE_PG_VECTOR


def build_retriever(chunks: List[OCRChunk]):
    """Build a LangChain retriever over OCR chunks.

    - If USE_PG_VECTOR=true and SUPABASE_URL is set with pgvector enabled,
      we use PGVector as the backing store (Supabase/Postgres).
    - Otherwise, we fall back to in-memory FAISS.
    """
    docs = [
        Document(
            page_content=c.text,
            metadata={
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page": c.page,
                "bbox": c.bbox,
            },
        )
        for c in chunks
    ]

    if OPENAI_API_KEY:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = FakeEmbeddings(size=32)

    # Try PGVector if requested
    if USE_PG_VECTOR and SUPABASE_URL:
        # Expect SUPABASE_URL like: postgresql://user:pass@host:5432/db
        # PGVector expects SQLAlchemy-style, so prepend '+psycopg2'
        conn_str = SUPABASE_URL
        if "://" in conn_str and "+psycopg2" not in conn_str:
            conn_str = conn_str.replace("://", "+psycopg2://", 1)

        collection_name = "ocr_icd_chunks"

        try:
            store = PGVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=collection_name,
                connection_string=conn_str,
            )
            return store.as_retriever()
        except Exception as e:
            # If pgvector not set up, fall back to FAISS
            print("PGVector setup failed, falling back to FAISS:", e)

    # Default: FAISS in-memory vector store
    store = FAISS.from_documents(docs, embeddings)
    return store.as_retriever()
