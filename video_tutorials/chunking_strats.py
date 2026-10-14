"""
Text Splitters and Chunking Strategies
Optimizing document chunks for RAG
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include: 
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50,
    separators=['\n\n','\n','. ',' ']
)

recursive_chunks = recursive_splitter.split_text(SAMPLE_TEXT)

print(f'Recursive chunks: {len(recursive_chunks)}')
for i, chunk in enumerate(recursive_chunks):
    print(f"\\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk)>100 else chunk)
    
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(SAMPLE_TEXT)

print(f"Semantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "..." if len(chunk)>100 else chunk)

# Create two vector stores - one for each chunking method
recursive_vectors = Chroma.from_texts(
    recursive_chunks,
    embeddings,
    collection_name = 'recursive_chunks' 
)


semantic_vectors = Chroma.from_texts(
    semantic_chunks,
    embeddings,
    collection_name = 'semantic_chunks' 
)

test_queries = [
    'What is Machine Learning?',
    'What are some common applications of Machine Learning?'
]

def test_retrieval(query,vectorstore,name):
    results = vectorstore.similarity_search(query,k=1)
    print(f'\\n{name} - Query: \"{query}\"')
    print(f'Retrieved: {results[0].page_content[:150]}...')
    return results[0].page_content

for query in test_queries:
    print('='*60)
    recursive_results = test_retrieval(query,recursive_vectors,'RECURSIVE')
    semantic_results = test_retrieval(query,semantic_vectors,'SEMANTIC')
    
def smart_chunker(
    text:str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500
) -> list[str]:
  """
  Production chunking with semantic as primary, recursive as fallback.
  """
  embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
  
  if use_semantic:
      try:
          chunker = SemanticChunker(
              embeddings,
              breakpoint_threshold_type = 'percentile',
              breakpoint_threshold_amount = 90
          )
          chunks = chunker.split_text(text)
    
          max_chunk_size = 2000
          if any(len(c)>max_chunk_size for c in chunks):
              return _recursive_fallback(text,fallback_chunk_size)
          return chunks
          
      except Exception as e:
          print(f'Semantic chunking failed: {e}, using fallback')
          return _recursive_fallback(text,fallback_chunk_size)
      
  return _recursive_fallback(text,fallback_chunk_size)

def _recursive_fallback(text:str,chunk_size:int)-> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=50
    )
    
    return splitter.split_text(text)


