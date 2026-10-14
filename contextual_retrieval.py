from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import time
import tiktoken

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def demonstrate_context_loss():
    """
    Show how chunking causes context loss.
    """

    print("=" * 60)
    print("THE PROBLEM: Chunks Lose Context")
    print("=" * 60)

    # Original document
    full_document = """
    ACME Corporation Annual Report 2025

    Company Overview:
    ACME Corporation was founded in 1994 in San Francisco. The company
    specializes in manufacturing industrial equipment for the mining sector.

    Financial Highlights:
    Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
    increase from the previous year. The company's profit margin improved
    to 18%, up from 14% in 2024.

    Future Outlook:
    The company plans to expand into renewable energy equipment in 2026.
    A new manufacturing facility will open in Austin, Texas. The company
    expects revenue growth of 20% in the coming fiscal year.
    """
    
    # Simulate chunking
    chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    print("\nOriginal Document: ACME Corporation Annual Report 2025")
    print("\nAfter chunking, we get isolated pieces like:")
    for i, chunk in enumerate(chunks, 1):
        print(f'\n  Chunk {i}: "{chunk}"')

    print("\n" + "-" * 60)
    print("THE PROBLEM:")
    print("-" * 60)
    print(
        """
    - Chunk 1: "The company" - Which company?
    - Chunk 2: "fiscal year 2025" - For what company?
    - Chunk 3: "The company plans" - Plans of what company?

    When a user asks "What is ACME's revenue?", semantic search might
    not match Chunk 2 because "ACME" doesn't appear in the chunk!
    """
    )
    
# ============================================================
# THE SOLUTION: CONTEXTUAL RETRIEVAL
# ============================================================

def add_contextual_prefix(
    chunk: str, full_document:str, document_title:str, llm: ChatOpenAI
    ) -> str:
    """
    Use LLM to generate a contextual prefix for a chunk.

    This is the core of Contextual Retrieval:
    - Give the LLM the full document and the specific chunk
    - Ask it to write a brief context that situates the chunk
    - Prepend this context to the chunk before embedding
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at adding context to document chunks.

                Given a document and a specific chunk from that document, write a SHORT
                context (1-2 sentences) that helps situate the chunk within the document.

                The context should include:
                - The document title/source if relevant
                - Key entities mentioned earlier in the document
                - Any important context that helps understand this chunk

                Keep it brief - just enough to disambiguate the chunk.
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
                        ]
                    )

    chain = prompt | llm
    response = chain.invoke(
        {"title": document_title, "document": full_document, "chunk": chunk}
    )

    return response.content
       
def demonstrate_contextual_retrieval():
    
    """
    Show how Contextual Retrieval solves the context loss problem.
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: Contextual Retrieval")
    print("=" * 60)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Full document for context
    full_document = """
    ACME Corporation Annual Report 2025

    Company Overview:
    ACME Corporation was founded in 1994 in San Francisco. The company
    specializes in manufacturing industrial equipment for the mining sector.

    Financial Highlights:
    Revenue for fiscal year 2025 reached $4.2 billion, representing a 15%
    increase from the previous year. The company's profit margin improved
    to 18%, up from 14% in 2024.

    Future Outlook:
    The company plans to expand into renewable energy equipment in 2026.
    A new manufacturing facility will open in Austin, Texas. The company
    expects revenue growth of 20% in the coming fiscal year.
    """

    document_title = "ACME Corporation Annual Report 2025"

    # Original chunks (without context)
    original_chunks = [
        "The company specializes in manufacturing industrial equipment for the mining sector.",
        "Revenue for fiscal year 2025 reached $4.2 billion, representing a 15% increase from the previous year.",
        "The company plans to expand into renewable energy equipment in 2026.",
    ]

    print("\nAdding contextual prefixes to each chunk...")
    print("-" * 60)
   
    contextualized_chunks = []
    for i, chunk in enumerate(original_chunks,1):
        print(f'\nProcessing chunk {i}...')
        
        # Generate contextual prefix
        context_prefix = add_contextual_prefix(
            chunk=chunk,
            full_document=full_document,
            document_title=document_title,
            llm=llm
        )
        
        # Combine prefix + original chunk
        contextualized = f"{context_prefix} {chunk}"
        contextualized_chunks.append(contextualized)

        print(f'  Original: "{chunk[:50]}..."')
        print(f'  Context:  "{context_prefix}"')
        print(f'  Combined: "{contextualized[:80]}..."')

    return original_chunks, contextualized_chunks

# ============================================================
# RETRIEVAL COMPARISON
# ============================================================    

def compare_retrieval(original_chunks:list,contextualized_chunks:list):
    """
    Compare retrieval quality between original and contextualized chunks.
    """
    
    print("\n" + "=" * 60)
    print("RETRIEVAL COMPARISON")
    print("=" * 60)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # create vector stores
    original_docs = [
        Document(page_content=c,metadata={'type':'original'}) for c in original_chunks
    ]
    
    context_docs = [
        Document(page_content=c,metadata={'type':'contextual'}) for c in contextualized_chunks
    ]
    
    vs_original = Chroma.from_documents(
        documents=original_docs, embedding=embeddings, collection_name="original_chunks"
    )

    vs_contextual = Chroma.from_documents(
        documents=context_docs,
        embedding=embeddings,
        collection_name="contextual_chunks",
    )
    
    # Test queries
    test_queries = [
        "What is ACME's revenue?",
        "What does ACME Corporation manufacture?",
        "What are ACME's expansion plans?",
    ]

    print("\nTesting retrieval with different queries:")
    print("-" * 60)
    
    for query in test_queries:
        print(f'\nQuesty: {query}')
        
        # Search Original
        orig_results = vs_original.similarity_search_with_score(query,k=1)
        orig_doc, orig_score = orig_results[0]
        
        # search contextual
        ctx_results = vs_contextual.similarity_search_with_score(query,k=1)
        ctx_doc, ctx_score = ctx_results[0]
        
        print(f"\n  Original chunks (score {orig_score:.3f}):")
        print(f'    "{orig_doc.page_content[:60]}..."')

        print(f"\n  Contextual chunks (score {ctx_score:.3f}):")
        print(f'    "{ctx_doc.page_content[:60]}..."')

        # Lower score = better match in Chroma
        if ctx_score < orig_score:
            improvement = ((orig_score - ctx_score) / orig_score) * 100
            print(f"\n  -> Contextual retrieval {improvement:.1f}% better match!")
        
    # Cleanup
    vs_original.delete_collection()
    vs_contextual.delete_collection()

# ============================================================
# PRODUCTION IMPLEMENTATION
# ============================================================

def create_contextual_chunks(
    documents: list[Document],
    llm: ChatOpenAI,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    
    """
    Production-ready function to create contextualized chunks.

    This is the function you'd use in a real RAG pipeline.
    Note: This adds latency and cost during indexing, but it's a one-time cost.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    contextualized_docs = []
    
    for doc in documents:
        # Get full document content and title
        full_content = doc.page_content
        doc_title = doc.metadata.get('title',doc.metadata.get("source","Unknown Document"))
       
        # Split into chunks
        chunks = text_splitter.split_text(full_content)
        
        # Contextualize each chunk 
        for i, chunk in enumerate(chunks):
            # Generate context prefix
            context_prefix = add_contextual_prefix(
                chunk = chunk,
                full_document = full_content,
                document_title = doc_title,
                llm = llm
            )     
        
        # Create new document with context
        contextualized_content = f"{context_prefix} {chunk}"

        contextualized_docs.append(
            Document(
                page_content=contextualized_content,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "original_chunk": chunk,
                    "context_prefix": context_prefix,
                },
            )
        )

    return contextualized_docs


        
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.2: CONTEXTUAL RETRIEVAL")
    print("=" * 60)
    print("\nAnthropic's technique to reduce retrieval failures by 67%")
    print("=" * 60)

    # 1. Show the problem
    demonstrate_context_loss()

    # 2. Show the solution
    original, contextualized = demonstrate_contextual_retrieval()

    # 3. Compare retrieval
    compare_retrieval(original, contextualized)

    # 4. Production demo
    #demo_production_pipeline()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
        1. Chunks lose context when split from documents
        2. Contextual Retrieval adds context BEFORE embedding
        3. Use LLM to generate 1-2 sentence context prefix
        4. One-time cost at indexing, permanent retrieval improvement
        5. Anthropic reports 67% fewer retrieval failures
        6. Combine with BM25 hybrid search for best results
        """
    )



