# load_books.py
import os
import re
from uuid import uuid4
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from supabase import create_client
from openai import OpenAI

load_dotenv()

# OpenAI client (new SDK style)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY must be set in environment to run loader")
client = OpenAI(api_key=openai_api_key)

# Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BOOKS_PATH = "homework/week2/books"
DOCUMENT_TYPE = "book"

# defaults for user metadata (avoid hardcoding 'naveen')
DEFAULT_USERNAME = os.environ.get("DEFAULT_USERNAME", "naveen")
DEFAULT_USER_ID = os.environ.get("DEFAULT_USER_ID", "naveen")


def read_book_text(book_folder):
    full_text = ""
    for root, dirs, files in os.walk(book_folder):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    full_text += f.read() + "\n"
            elif file.endswith(".html"):
                with open(path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    full_text += soup.get_text(separator="\n") + "\n"
            else:
                # ignore images (jpg/png) — titles in image should be handled externally
                continue
    return full_text


def split_into_chapters(text):
    chapter_regex = r"(CHAPTER\s+[A-Z0-9]+.*)"
    parts = re.split(chapter_regex, text, flags=re.IGNORECASE)
    chapters = []
    i = 1
    while i < len(parts):
        title = parts[i].strip()
        content = parts[i + 1].strip()
        chapters.append((title, content))
        i += 2
    return chapters


def split_into_paragraphs(chapter_text):
    paragraphs = [p.strip() for p in chapter_text.split("\n") if p.strip()]
    return paragraphs


def get_embedding(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def upload_books_to_supabase():
    for book_folder in os.listdir(BOOKS_PATH):
        folder_path = os.path.join(BOOKS_PATH, book_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing book folder: {book_folder}")
        book_text = read_book_text(folder_path)
        if not book_text.strip():
            print(f"⚠️ No text found in {book_folder}, skipping.")
            continue

        chapters = split_into_chapters(book_text)
        if len(chapters) == 0:
            chapters = [("Full Book", book_text)]

        total = 0
        for chapter_title, chapter_content in chapters:
            paragraphs = split_into_paragraphs(chapter_content)
            for paragraph_number, paragraph in enumerate(paragraphs, start=1):
                if len(paragraph) < 20:
                    continue
                embedding = get_embedding(paragraph)
                metadata = {
                    "source": "local_books_folder",
                    "book_folder": book_folder,
                    "chapter_title": chapter_title
                }
                row = {
                    "id": str(uuid4()),
                    "embedding": embedding,
                    "context": paragraph,
                    "user_id": DEFAULT_USER_ID,
                    "username": DEFAULT_USERNAME,
                    "document_type": DOCUMENT_TYPE,
                    "document_id": book_folder,
                    "chapter_title": chapter_title,
                    "paragraph_number": paragraph_number,
                    "metadata": metadata
                }
                supabase.table("rag_content").insert(row).execute()
                total += 1
        print(f"Uploaded {total} chunks for {book_folder}")


if __name__ == "__main__":
    upload_books_to_supabase()
