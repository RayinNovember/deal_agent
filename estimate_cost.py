"""
Estimate OpenAI cost of (contextual) chunking + embedding for a set of PDFs
BEFORE making any API calls. Loading, splitting and token counting are all local/free.

Usage:
    uv run estimate_cost.py "path/to/folder1" "path/to/folder2"
"""

import sys
from pathlib import Path

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Settings: match these to your real pipeline ----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
PROMPT_OVERHEAD = 200      # tokens for the system message + template wording
PREFIX_TOKENS = 60         # expected output length of each contextual prefix

# USD per 1M tokens -- check https://developers.openai.com/api/docs/pricing
PRICE_LLM_INPUT = 0.15     # gpt-4o-mini input
PRICE_LLM_CACHED = 0.075   # gpt-4o-mini cached input
PRICE_LLM_OUTPUT = 0.60    # gpt-4o-mini output
PRICE_EMBED = 0.02         # text-embedding-3-small
# -----------------------------------------------------------------------------

try:
    # gpt-4o-mini uses the o200k_base encoding (older tiktoken versions don't know the model name)
    enc = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(enc.encode(text))
except Exception as e:
    # tiktoken downloads its encoding file on first use; a firewall/proxy can block that
    print(f"[warn] tiktoken unavailable ({type(e).__name__}); using ~4 chars/token estimate")
    count = lambda text: len(text) // 4
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def estimate_pdf(pdf_path: Path) -> dict:
    pages = PyPDFLoader(str(pdf_path)).load()
    doc_tokens = count("\n".join(p.page_content for p in pages))
    chunks = splitter.split_documents(pages)
    chunk_tokens = [count(c.page_content) for c in chunks]
    n = len(chunks)

    # Contextual retrieval: one LLM call per chunk, each sending doc + prompt + chunk
    per_call_prefix = doc_tokens + PROMPT_OVERHEAD             # identical part (cacheable)
    llm_input = n * per_call_prefix + sum(chunk_tokens)
    llm_output = n * PREFIX_TOKENS

    uncached = llm_input * PRICE_LLM_INPUT / 1e6
    # Best case caching: first call full price, later calls reuse the cached prefix
    cached = ((per_call_prefix + sum(chunk_tokens)) * PRICE_LLM_INPUT
              + (n - 1) * per_call_prefix * PRICE_LLM_CACHED) / 1e6
    output = llm_output * PRICE_LLM_OUTPUT / 1e6

    embed_plain = sum(chunk_tokens) * PRICE_EMBED / 1e6
    embed_ctx = (sum(chunk_tokens) + llm_output) * PRICE_EMBED / 1e6

    return {
        "file": pdf_path.name, "pages": len(pages), "doc_tokens": doc_tokens, "chunks": n,
        "plain": embed_plain,
        "ctx_low": cached + output + embed_ctx,
        "ctx_high": uncached + output + embed_ctx,
    }


def main(folders):
    pdfs = [p for f in folders for p in sorted(Path(f).rglob("*.pdf"))]
    if not pdfs:
        print("No PDFs found.")
        return

    rows = [estimate_pdf(p) for p in pdfs]

    print(f"\n{'File':40} {'Pages':>5} {'Tokens':>8} {'Chunks':>6} {'Plain $':>9} {'Contextual $':>17}")
    print("-" * 90)
    for r in rows:
        print(f"{r['file'][:40]:40} {r['pages']:>5} {r['doc_tokens']:>8,} {r['chunks']:>6} "
              f"{r['plain']:>9.4f} {r['ctx_low']:>8.4f}-{r['ctx_high']:.4f}")
    print("-" * 90)
    print(f"{'TOTAL':40} {sum(r['pages'] for r in rows):>5} "
          f"{sum(r['doc_tokens'] for r in rows):>8,} {sum(r['chunks'] for r in rows):>6} "
          f"{sum(r['plain'] for r in rows):>9.4f} "
          f"{sum(r['ctx_low'] for r in rows):>8.4f}-{sum(r['ctx_high'] for r in rows):.4f}")
    print("\nPlain = chunk + embed only. Contextual = LLM prefix per chunk + embed "
          "(range = with / without prompt caching).")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
