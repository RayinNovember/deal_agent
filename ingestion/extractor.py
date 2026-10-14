# ingestion/extractor.py
import json
import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Switch to AzureChatOpenAI at M10 — see roadmap M10 migration note
# llm = AzureChatOpenAI(
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     temperature=0,
# )
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

EXTRACTION_WINDOW = 2000  # chars — enough to cover the header and first section


def extract_document_metadata(doc: dict) -> dict:
    """
    One LLM call per document. Takes your existing doc dict from
    pdf_loader_pymupdf4llm() and returns structured metadata to attach
    to every chunk from that document.
    """
    first_chunk = doc["full_text"][:EXTRACTION_WINDOW]
    source_path = doc["file_path"]

    prompt = f"""You are extracting structured metadata from a real estate legal document.

Document path: {source_path}

First portion of document:
{first_chunk}

Extract and return ONLY valid JSON with these exact fields:
{{
  "doc_type": "PSA | loan_agreement | lease | amendment | underwriting_model | correspondence | other",
  "effective_date": "YYYY-MM-DD or null",
  "deal_name": "property or deal name as a short string, or null",
  "parties": ["list of party names mentioned"],
  "amendment_number": "integer if this is a numbered amendment, else null",
  "parent_document_id": "short reference to the original document this amends, or null",
  "summary": "2-3 sentence plain English description of this document's purpose",
  "key_financial_terms": {{
    "purchase_price": "dollar amount as string, or null",
    "loan_amount": "dollar amount as string, or null",
    "cap_rate": "percentage as string, or null"
  }}
}}

Return only the JSON object. No explanation, no markdown fences."""

    result = llm.invoke(prompt).content.strip()

    # Strip markdown fences if the model adds them anyway
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result.removeprefix("json").strip()
        result = result.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print(f"  [extractor] JSON parse failed for {doc['file_name']} — using path fallback")
        return _fallback(doc)


def _fallback(doc: dict) -> dict:
    """Path and title-based fallback when LLM extraction fails."""
    return {
        "doc_type": _guess_doc_type(doc["file_path"], doc["title"]),
        "effective_date": None,
        "deal_name": None,
        "parties": [],
        "amendment_number": None,
        "parent_document_id": None,
        "summary": None,
        "key_financial_terms": {},
         "_fallback": True,   # <-- flag
    }


def _guess_doc_type(path: str, title: str) -> str:
    text = (path + " " + (title or "")).lower()
    if "psa" in text or "purchase" in text or "sale" in text:
        return "PSA"
    if "loan" in text or "mortgage" in text or "note" in text:
        return "loan_agreement"
    if "lease" in text:
        return "lease"
    if "amend" in text:
        return "amendment"
    return "other"