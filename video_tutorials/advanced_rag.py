"""
Advanced RAG Patterns
Multi-query, self-query, compression, hybrid search
"""
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging

load_dotenv()

# Sample knowledge base for demos
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

def demo_parent_document_retriever():
    """Parent Document Retriever: small chunks for search, large for context."""

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)
    # Long document to demonstrate parent/child splitting
    long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
        """,
        metadata={"source": "ai_agents_guide.md"},
    )
    

    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    
    # Storage
    vectorstore = Chroma(
        collection_name = 'parent_child_demo',
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
    )
    
    store = InMemoryStore()
    
    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    # Add document
    retriever.add_documents([long_doc])
    
    # Search
    query = 'What is LangGraph used for?'
    
    print(f'\nQuery: {query}')
    
    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=1)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")



def create_base_vectorstore():
    """Create a basic vector store for demos."""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

def demo_contextual_compression():
    """Contextual Compression extracts only relevant parts."""

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant content from documents")
    print("=" * 60)
    
    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    
    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    
    # Storage
    vectorstore = Chroma(
        collection_name = 'parent_child_demo',
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
    )
    
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    
    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    docs = compression_retriever.invoke("What is the late fee policy?")


def demo_multi_query_retriever():
    """Multi-Query Retriever generates multiple query perspectives."""

    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("Generates multiple perspectives on your question")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), llm=llm
    )

    query = "What tools can I use to build AI applications?"

    print(f"\nOriginal Query: {query}")
    print("\nThe retriever will generate multiple query variations...")
    print("(Check INFO logs above for generated queries)\n")

    # Retrieve documents
    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )
    

 























if __name__ =='__main__':
    demo_parent_document_retriever() 