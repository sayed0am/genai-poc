#!/usr/bin/env python3
"""
Simple interface for querying processed documents using the vector index.
"""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
# from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine

from sqlalchemy import make_url

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration (must match the ones used during processing)
GEN_MODEL = Ollama(model="qwen3:4b-instruct-2507-q8_0", temperature=0.7, request_timeout=160.0 , keep_alive="10m", context_window=2048)
EMBED_MODEL = OllamaEmbedding(model_name="nomic-embed-text")

# Database configuration (must match the ones used during indexing)
CONNECTION_STRING = "postgresql://postgres:password@localhost:5432"
DB_NAME = "vector_db"

# Query configuration
DEFAULT_TOP_K = 3
DEFAULT_NUM_QUERIES = 1  # Set to 1 to disable query generation, increase for query expansion

# ============================================================================
# INDEX LOADING
# ============================================================================

def load_existing_index(connection_string=CONNECTION_STRING, db_name=DB_NAME):
    """
    Load existing vector index from PostgreSQL database.
    
    Args:
        connection_string: PostgreSQL connection string
        db_name: Name of the database
    
    Returns:
        VectorStoreIndex: Loaded vector index
    """
    print("Loading existing vector index...")
    
    # Create vector store connection
    url = make_url(connection_string)
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name="proc_docs",
        embed_dim=768,
        hybrid_search=True,
        text_search_config="english",
        indexed_metadata_keys={("context", "text")},
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    
    # Load the index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=EMBED_MODEL
    )
    
    print("✅ Vector index loaded successfully")
    return index

# ============================================================================
# QUERY ENGINE SETUP
# ============================================================================

def create_retriever(index, top_k=DEFAULT_TOP_K, num_queries=DEFAULT_NUM_QUERIES):
    """
    Create a hybrid retrieval (vector + text search).
    
    Args:
        index: Vector index
        top_k: Number of top results to retrieve
        num_queries: Number of queries for fusion (1 = no expansion)
    
    Returns:
        Retriever
    """
    
    # Create vector retriever (semantic search)
    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=5,
    )
    
    # Create text retriever (keyword search)
    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse",
        similarity_top_k=5,
    )
    
    # Create fusion retriever (combines both approaches)
    retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, text_retriever],
        llm=GEN_MODEL,
        similarity_top_k=top_k,
        num_queries=num_queries,
        mode="relative_score",
        use_async=True,
    )
    
    print("Retriever ready")
    return retriever


def create_contextualized_chunk(node, chunk_index ,include_metadata=True, include_context=True):
    """
    Create a single contextualized chunk combining content, metadata, and context.
    
    Args:
        node: Source node from query results
        include_metadata: Whether to include file name and page number
        include_context: Whether to include the contextual information
    
    Returns:
        str: Formatted contextualized chunk
        
    Example output:
        [Source: research_paper.pdf, Page: 3]
        [Context: This section discusses the methodology used in the machine learning experiment]
        [Content: The research team implemented a novel neural network architecture that achieved 
        95% accuracy on the test dataset. The model was trained using supervised learning techniques...]
    """
    parts = []
    
    # Extract metadata
    metadata = node.metadata
    file_name = metadata.get("file_name", "Unknown File")
    
    # Extract page number from nested metadata structure
    page_no = "Unknown Page"
    try:
        if "doc_items" in metadata and len(metadata["doc_items"]) > 0:
            if "prov" in metadata["doc_items"][0] and len(metadata["doc_items"][0]["prov"]) > 0:
                page_no = metadata["doc_items"][0]["prov"][0].get("page_no", "Unknown Page")
    except (KeyError, IndexError, TypeError):
        pass
    
    # Extract context
    context = metadata.get("context", "")
    
    if include_metadata:
        parts.append(f"[Document Chunk:{chunk_index + 1}, Source: {file_name}, Page: {page_no}]")
    
    if include_context and context:
        parts.append(f"[Context: {context}]")
    
    # Add the main content
    parts.append(f"[Content: {node.text.strip()}]")
    
    return "\n".join(parts)

def get_all_contextualized_chunks(retriever, question, max_chunks=10):
    """
    Retrieve raw chunks and return them contextualized.
    
    Args:
        retriever: Configured retriever (from query engine)
        question: Question to ask
        max_chunks: Maximum number of chunks to return
    
    Returns:
        list: List of contextualized chunks
    """
    print(f"🔍 Retrieving contextualized chunks for: {question}")
    
    # Get raw retrieval results
    nodes = retriever.retrieve(question)
    
    # Create contextualized chunks
    contextualized_chunks = []
    
    for i, node in enumerate(nodes[:max_chunks]):
        chunk = create_contextualized_chunk(node, chunk_index=i)
        contextualized_chunks.append(chunk)
    
    return contextualized_chunks


# ============================================================================
# def document_retrieval_tool(query: str, max_chunks=5) -> str:    
#     # Load index and create query engine
#     index = load_existing_index()
#     retriever = create_retriever(index)
    
#     context_chunks = get_all_contextualized_chunks(retriever, query.strip(), max_chunks=max_chunks)

#     seperator = "\n\n"+"--"*50+"\n\n"
#     context = seperator.join(context_chunks)

#     return context


class DocumentRetrievalInput(BaseModel):
    """Input schema for DocumentRetrievalTool."""
    query: str = Field(..., description="The search query to find relevant document chunks")
    max_chunks: int = Field(default=5, description="Maximum number of document chunks to retrieve (default: 5)")

class DocumentRetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = """
    Searches through a vector database of processed documents to find relevant chunks based on a query.
    Uses hybrid search (semantic + keyword) to retrieve the most relevant document chunks.
    Returns contextualized chunks with source information, page numbers, and content.
    """
    args_schema: Type[BaseModel] = DocumentRetrievalInput
    
    def _run(self, query: str, max_chunks: int = 5) -> str:
        """
        Execute the document retrieval.
        
        Args:
            query: The search query
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            str: Formatted document chunks with context
        """
        try:
            # Load index and create query engine
            index = load_existing_index()
            retriever = create_retriever(index)
            
            # Get contextualized chunks
            context_chunks = get_all_contextualized_chunks(
                retriever, 
                query.strip(), 
                max_chunks=max_chunks
            )
            
            # Join chunks with separator
            separator = "\n\n" + "--" * 50 + "\n\n"
            context = separator.join(context_chunks)
            
            return context
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
        

# print(document_retrieval_tool("What is the maximum lump sum financial reward, in Dirhams, that a military retiree in the 'First' main grade can receive upon appointment?", max_chunks=3)
# )
