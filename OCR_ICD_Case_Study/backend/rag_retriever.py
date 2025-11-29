# rag_retriever.py
# ---------------------------------------------------------
# MOCK retriever for demo without OpenAI embeddings,
# without FAISS, and without PGVector.
# ---------------------------------------------------------

from typing import List

from .schemas import OCRChunk


def build_retriever(chunks: List[OCRChunk]):
    """
    Very simple keyword-based retriever.
    Compatible with LangChain's retriever interface
    by implementing both:
        - get_relevant_documents(query)
        - invoke(query)

    No OpenAI, no embeddings, no FAISS, no PGVector.
    Works fully offline.
    """

    entries = [
        {
            "text": c.text,
            "metadata": {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page": c.page,
                "bbox": c.bbox,
            },
        }
        for c in chunks
    ]

    class SimpleRetriever:
        def get_relevant_documents(self, query: str):
            """Return chunks whose text contains any of the keywords."""
            if not query:
                return []

            keywords = query.lower().split()

            results = []
            for entry in entries:
                text = entry["text"].lower()
                if any(k in text for k in keywords):
                    results.append(
                        {
                            "page_content": entry["text"],
                            "metadata": entry["metadata"],
                        }
                    )
            return results

        # ⭐⭐⭐ This FIX makes LangChain compatible
        def invoke(self, query: str):
            return self.get_relevant_documents(query)

    return SimpleRetriever()
