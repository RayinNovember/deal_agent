from ingestion.ingest import pdf_loader_pymupdf4llm
from ingestion.extractor import extract_document_metadata

PDF_FOLDERS = ['docs/time_series']  # or point at your contracts folder

docs = pdf_loader_pymupdf4llm(PDF_FOLDERS)

for doc in docs[:4]:
    print(f"\n{'='*60}")
    print(f"FILE: {doc['file_name']}")
    metadata = extract_document_metadata(doc)
    for k, v in metadata.items():
        print(f"  {k}: {v}")