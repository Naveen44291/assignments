-- rag_content.sql
-- Enable required extensions
create extension if not exists vector;
create extension if not exists pgcrypto;

-- Drop old table (optional)
drop table if exists public.rag_content cascade;

-- Create the main RAG content table
create table public.rag_content (
    id uuid primary key default gen_random_uuid(),
    embedding vector(1536),
    context text,
    user_id text,
    username text,
    document_type text,
    document_id text,
    chapter_title text,
    paragraph_number int,
    created_at timestamptz default now(),
    metadata jsonb
);

-- Vector index for similarity search (ivfflat with cosine)
create index if not exists idx_rag_content_embedding on public.rag_content
using ivfflat (embedding vector_cosine)
with (lists = 100);

-- GIN index for metadata lookups (if you use metadata keys later)
create index if not exists idx_rag_content_metadata_gin on public.rag_content using gin (metadata);

-- RPC: match_rag (supports metadata filtering)
create or replace function public.match_rag(
    query_embedding vector(1536),
    match_count int,
    query_document_types text[] default null,
    filter_username text default null,
    filter_user_id text default null,
    filter_document_id text default null,
    filter_chapter_title text default null,
    min_paragraph int default null,
    max_paragraph int default null
)
returns table (
    id uuid,
    context text,
    document_type text,
    document_id text,
    chapter_title text,
    paragraph_number int,
    username text,
    user_id text,
    similarity float
)
language sql stable as $$
    select
        rc.id,
        rc.context,
        rc.document_type,
        rc.document_id,
        rc.chapter_title,
        rc.paragraph_number,
        rc.username,
        rc.user_id,
        1 - (rc.embedding <=> query_embedding) as similarity
    from public.rag_content rc
    where
        (query_document_types is null or rc.document_type = any(query_document_types))
        and (filter_username is null or rc.username = filter_username)
        and (filter_user_id is null or rc.user_id = filter_user_id)
        and (filter_document_id is null or rc.document_id = filter_document_id)
        and (filter_chapter_title is null or rc.chapter_title = filter_chapter_title)
        and (min_paragraph is null or rc.paragraph_number >= min_paragraph)
        and (max_paragraph is null or rc.paragraph_number <= max_paragraph)
    order by rc.embedding <=> query_embedding
    limit match_count;
$$;
