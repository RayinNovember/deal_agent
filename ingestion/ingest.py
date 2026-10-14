import hashlib
import re
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from ingestion.chunk_enrichment import enrich_static, enrich_contextual

load_dotenv()

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
PDF_FOLDERS = ['docs/time_series']
FILE_KEYS = ("title", "file_name", "file_path", "folder_name", "folder_path", "relative_path")
KEEP = ("title", "author", "creationDate", "page_count", "page_number")
MIN_CHARS = 50
ENRICHMENT_MODE = "static"   # "static" | "contextual"

# ============================================================
# STEP 1: LOAD
# ============================================================

def pdf_loader_pymupdf4llm(folders: list[str]) -> list[dict]:
    docs = []
    for folder in folders:
        root = Path(folder)
        for pdf_path in sorted(root.rglob("*.pdf")):
            md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)

            title = md_pages[0]["metadata"].get("title") or pdf_path.stem

            file_meta = {
                "title": title,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path.resolve()),
                "folder_name": pdf_path.parent.name,
                "folder_path": str(pdf_path.parent.resolve()),
                "relative_path": str(pdf_path.relative_to(root)),
            }

            pages = []
            for p in md_pages:
                meta = {k: p["metadata"][k] for k in KEEP if p["metadata"].get(k) not in (None, "")}
                meta.update(file_meta)
                pages.append(Document(page_content=p["text"], metadata=meta))

            docs.append({
                **file_meta,
                "pages": pages,
                "full_text": "\n".join(p.page_content for p in pages),
            })
            print(f"Loaded {pdf_path.name} from '{file_meta['folder_name']}': "
                  f"{len(pages)} pages, {len(docs[-1]['full_text']):,} chars, title='{title}'")
    return docs


# ============================================================
# STEP 2: CHUNKING
# pymupdf4llm converts PDF → markdown text (with # headings, **bold**, <sup> tags)
#   ↓
# chunk_by_heading()   splits on the # markers → sections with messy metadata
#   ↓
# clean_section()      strips the markdown/HTML noise from the heading labels
#   ↓
# chunk_by_size_header() re-attaches file metadata, size-splits, drops tiny scraps
#   ↓
# enrich_chunks()      adds context prefix, chunk ID, and LLM-extracted metadata
# ============================================================


# ============================================================
# STEP 2a: SPLIT BY MARKDOWN HEADINGS
# ============================================================

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5")],
    strip_headers=False,
)

def chunk_by_heading(doc: dict) -> list[Document]:
    sections = header_splitter.split_text(doc['full_text'])
    print(f"\n[2a] {doc['file_name']}: {len(sections)} sections")
    for s in sections[:8]:
        print(s.metadata, f"({len(s.page_content):,} chars)")
    return sections


# ============================================================
# STEP 2b: CLEAN UP HEADINGS AND LABEL SECTIONS
# ============================================================

def clean_heading(text: str) -> str:
    """Strip markdown/HTML formatting: **bold**, _italic_, <sup>…</sup>."""
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"[*_]", "", text)
    return " ".join(text.split())

def clean_section(sections: list[Document]) -> list[Document]:
    """Replace h1..h5 with the nearest heading only."""
    for s in sections:
        nearest = list(s.metadata.values())[-1] if s.metadata else ""
        s.metadata = {"section": clean_heading(nearest)}
    print(f"\n[2b] {len(sections)} labelled sections")
    for s in sections[:8]:
        print(s.metadata, f"({len(s.page_content):,} chars)")
    return sections


# ============================================================
# STEP 2c: SPLIT BY SIZE (CHUNK_SIZE, CHUNK_OVERLAP)
# ============================================================

size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def chunk_by_size(doc: dict) -> list[Document]:
    chunks = size_splitter.split_documents(doc["pages"])
    print(f"\n[2c] {doc['file_name']}: {len(doc['pages'])} pages -> {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n--- page {c.metadata['page_number']} ---")
        print(c.page_content)
    return chunks


# ============================================================
# STEP 2d: CHUNK BY SIZE WITH SECTION LABELS
# ============================================================

def chunk_by_size_header(doc: dict) -> list[Document]:
    clean_sections = clean_section(chunk_by_heading(doc))

    file_meta = {k: doc[k] for k in FILE_KEYS}
    for sec in clean_sections:
        sec.metadata.update(file_meta)

    section_chunks = size_splitter.split_documents(clean_sections)
    final_chunks = [c for c in section_chunks if len(c.page_content.strip()) >= MIN_CHARS]

    print(f"\n[2d] {doc['file_name']}: {len(clean_sections)} sections -> {len(final_chunks)} chunks")
    return final_chunks


# ============================================================
# STEP 2e: ENRICH CHUNKS WITH LLM METADATA + CONTEXT PREFIX
# Takes the heading+size chunks and stamps each one with:
#   - deterministic chunk ID
#   - context prefix prepended to the embedding text
#   - all LLM-extracted document metadata
# Output is a list of plain dicts ready for Qdrant upsert.
# ============================================================



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from ingestion.extractor import extract_document_metadata

    docs = pdf_loader_pymupdf4llm(PDF_FOLDERS)
    print(f"\nTotal: {len(docs)} PDFs")
    doc = docs[0]

    # --- 2a: heading split ---
    sections = chunk_by_heading(doc)

    # --- 2b: clean section labels ---
    clean_sections = clean_section(sections)

    # --- 2c: size split only ---
    size_chunks = chunk_by_size(doc)

    # --- 2d: heading + size ---
    chunks = chunk_by_size_header(doc)

    # --- 2e: enrich ---
    meta = extract_document_metadata(doc)      # <-- add this

    if ENRICHMENT_MODE == "contextual":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        enriched = enrich_contextual(chunks, doc, meta, llm)
    else:
        enriched = enrich_static(chunks, doc, meta)

    # size check
    sizes = [len(c["text_for_display"]) for c in enriched]
    print(f"\nChunk sizes: min {min(sizes)}, max {max(sizes)}, avg {sum(sizes) // len(sizes)}")

    # sample chunks
    for i in (0, len(enriched) // 2, -1):
        c = enriched[i]
        print(f"\n--- chunk {i} | section: '{c['section']}' | doc_type: '{c['doc_type']}' ---")
        print(c["text_for_display"][:300])
        print(f"  embedding prefix: {c['text_for_embedding'][:80]}...")

    print("\nSample enriched chunk:", {k: v for k, v in enriched[len(enriched) // 2].items() if k != "text_for_embedding"})