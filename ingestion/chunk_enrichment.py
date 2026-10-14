"""
ingestion/chunk_enrichment.py

Two strategies for enriching chunks before embedding:

  enrich_static()      — one LLM call per document (fast, cheap)
                         Prepends [doc_type | deal_name | effective_date] to
                         every chunk so the vector carries document identity.

  enrich_contextual()  — one LLM call per chunk (slower, richer)
                         Anthropic's Contextual Retrieval approach: the LLM
                         reads the full document + the specific chunk and writes
                         a bespoke 1-2 sentence prefix that situates the chunk
                         within the document.

Both return the same list[dict] shape so pipeline.py can treat them identically.
Switch between them with ENRICHMENT_MODE in ingest.py.
"""

import hashlib
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# SHARED HELPERS
# ============================================================

def _make_chunk_id(file_path: str, index: int) -> str:
    """Deterministic 16-char ID from file path + chunk position.
    Re-ingesting the same file produces the same IDs → Qdrant upserts
    overwrite instead of duplicating.
    """
    return hashlib.sha256(f"{file_path}::{index}".encode()).hexdigest()[:16]


def _base_record(chunk: Document, doc: dict, meta: dict, index: int, total: int) -> dict:
    """Fields shared by both enrichment strategies."""
    return {
        # Identity
        "id": _make_chunk_id(doc["file_path"], index),
        "chunk_index": index,
        "total_chunks": total,
        # Chunk-level metadata (from heading split)
        "section": chunk.metadata.get("section"),
        # File metadata
        "source_path": doc["file_path"],
        "file_name": doc["file_name"],
        "title": doc["title"],
        # LLM-extracted document metadata (same on every chunk from this doc)
        "doc_type": meta.get("doc_type"),
        "effective_date": meta.get("effective_date"),
        "deal_name": meta.get("deal_name"),
        "parties": meta.get("parties", []),
        "amendment_number": meta.get("amendment_number"),
        "parent_document_id": meta.get("parent_document_id"),
        "summary": meta.get("summary"),
        "key_financial_terms": meta.get("key_financial_terms", {}),
        "extraction_fallback": meta.get("_fallback", False),
    }


# ============================================================
# STRATEGY 1: STATIC PREFIX
# One LLM call per document (already done by extractor.py).
# Fast and cheap — good for development and initial production.
# ============================================================

def enrich_static(chunks: list[Document], doc: dict, meta: dict) -> list[dict]:
    """
    Prepend a fixed [doc_type | deal_name | effective_date] prefix to every
    chunk's embedding text. The prefix is the same for all chunks in a document.

    Cost: 0 extra LLM calls (metadata already extracted by extractor.py).
    """
    context_prefix = (
        f"[{meta.get('doc_type') or 'unknown'} | "
        f"{meta.get('deal_name') or 'unknown deal'} | "
        f"{meta.get('effective_date') or 'date unknown'}] "
    )

    total = len(chunks)
    enriched = []
    for i, chunk in enumerate(chunks):
        record = _base_record(chunk, doc, meta, i, total)
        record["text_for_display"] = chunk.page_content
        record["text_for_embedding"] = context_prefix + chunk.page_content
        record["enrichment_strategy"] = "static"
        enriched.append(record)

    print(f"\n[enrich_static] {doc['file_name']}: {total} chunks "
          f"(fallback={meta.get('_fallback', False)})")
    return enriched


# ============================================================
# STRATEGY 2: CONTEXTUAL PREFIX (Anthropic Contextual Retrieval)
# One LLM call per chunk — richer but significantly more expensive.
# The LLM reads the full document + the chunk and writes a bespoke
# 1-2 sentence prefix that situates the chunk in context.
#
# Use when:
#   - Static prefix retrieval quality is not good enough
#   - You have Azure OpenAI provisioned (keeps cost in-house)
#   - Document count is manageable (100s not 10,000s)
# ============================================================

_CONTEXTUAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at adding context to document chunks.

Given a document and a specific chunk from that document, write a SHORT
context (1-2 sentences) that helps situate the chunk within the document.

The context should include:
- The document title/source if relevant
- Key entities mentioned earlier in the document
- Any important context that helps understand this chunk

Keep it brief — just enough to disambiguate the chunk.
Output ONLY the contextual prefix, nothing else.""",
    ),
    (
        "human",
        """Document Title: {title}

Full Document:
{document}

Chunk to contextualize:
{chunk}

Write a brief contextual prefix for this chunk:""",
    ),
])


def _generate_contextual_prefix(
    chunk_text: str,
    full_text: str,
    title: str,
    llm: ChatOpenAI,
) -> str:
    """Call the LLM to generate a bespoke prefix for one chunk."""
    chain = _CONTEXTUAL_PROMPT | llm
    response = chain.invoke({
        "title": title,
        "document": full_text,
        "chunk": chunk_text,
    })
    return response.content.strip()


def enrich_contextual(
    chunks: list[Document],
    doc: dict,
    meta: dict,
    llm: ChatOpenAI,
) -> list[dict]:
    """
    Generate a bespoke contextual prefix for each chunk using the full document
    as context. One LLM call per chunk.

    Cost: len(chunks) LLM calls — use gpt-4o-mini to keep this affordable,
    or Azure OpenAI in production to keep data in-house.
    """
    total = len(chunks)
    enriched = []

    for i, chunk in enumerate(chunks):
        print(f"  [enrich_contextual] {doc['file_name']} chunk {i+1}/{total}...", end="\r")

        prefix = _generate_contextual_prefix(
            chunk_text=chunk.page_content,
            full_text=doc["full_text"],
            title=doc["title"],
            llm=llm,
        )

        record = _base_record(chunk, doc, meta, i, total)
        record["text_for_display"] = chunk.page_content
        record["text_for_embedding"] = prefix + " " + chunk.page_content
        record["contextual_prefix"] = prefix          # stored separately for inspection
        record["enrichment_strategy"] = "contextual"
        enriched.append(record)

    print(f"\n[enrich_contextual] {doc['file_name']}: {total} chunks enriched")
    return enriched