from llama_index.core import SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import TextNode

from sqlalchemy import make_url
from datetime import datetime
import json
import os
import re
import psycopg2

# ============================================================================
# CONFIGURATION
# ============================================================================

GEN_MODEL = Ollama(model="qwen3:4b-instruct-2507-q8_0", temperature=0.7, keep_alive=True, context_window=2048)
EMBED_MODEL = OllamaEmbedding(model_name="nomic-embed-text")

# Document processing settings
DOCS_DIR = "docs"
OUTPUT_DIR = "contextualized_nodes"
BATCH_SIZE = 10
MAX_CONTEXT_TOKENS = 1500

# Database settings
CONNECTION_STRING = "postgresql://postgres:password@localhost:5432"
DB_NAME = "vector_db"

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def get_surrounding_pages_context(documents, nodes, chunk_index, max_tokens=1500):
    """
    Get context from the page before and after the current chunk.
    
    Args:
        documents: List of documents
        nodes: List of nodes from node parser
        chunk_index: Index of the current chunk
        max_tokens: Maximum tokens to use for context (leave room for prompt)
    
    Returns:
        dict: Contains context pages and metadata
    """
    current_node = nodes[chunk_index]
    current_page = current_node.metadata["doc_items"][0]["prov"][0]["page_no"]
    
    # Get all nodes and their page numbers for context
    page_content = {}
    
    # Collect all chunks by page number
    for i, node in enumerate(nodes):
        try:
            page_no = node.metadata["doc_items"][0]["prov"][0]["page_no"]
            if page_no not in page_content:
                page_content[page_no] = []
            page_content[page_no].append(node.text)
        except (KeyError, IndexError):
            # Skip nodes without proper page metadata
            continue
    
    # Get surrounding pages
    prev_page = current_page - 1
    next_page = current_page + 1
    
    context_parts = []
    
    # Add previous page if exists
    if prev_page in page_content:
        prev_text = "\n".join(page_content[prev_page])
        context_parts.append(f"[Page {prev_page}]\n{prev_text}")
    
    # Add current page
    if current_page in page_content:
        current_text = "\n".join(page_content[current_page])
        context_parts.append(f"[Page {current_page}]\n{current_text}")
    
    # Add next page if exists
    if next_page in page_content:
        next_text = "\n".join(page_content[next_page])
        context_parts.append(f"[Page {next_page}]\n{next_text}")
    
    # Combine context and truncate if necessary
    full_context = "\n\n".join(context_parts)
    
    # Simple token estimation (roughly 4 chars per token)
    estimated_tokens = len(full_context) // 4
    
    if estimated_tokens > max_tokens:
        # Truncate context to fit within limits
        target_chars = max_tokens * 4
        full_context = full_context[:target_chars] + "...\n[Context truncated due to length]"
    
    return {
        "context": full_context,
        "current_page": current_page,
        "pages_included": [p for p in [prev_page, current_page, next_page] if p in page_content],
        "estimated_tokens": len(full_context) // 4
    }


def create_contextual_prompt(chunk_text, surrounding_context):
    """
    Create the contextual retrieval prompt with surrounding pages.
    """
    prompt_template = """<document_context>
{SURROUNDING_CONTEXT}
</document_context>

Here is the chunk we want to situate within the document context above:
<chunk>
{CHUNK_CONTENT}
</chunk>

Please give a short succinct context to situate this chunk within the surrounding document pages for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    
    return prompt_template.format(
        SURROUNDING_CONTEXT=surrounding_context,
        CHUNK_CONTENT=chunk_text
    )


def clean_llm_response(response):
    """
    Clean the LLM response by removing assistant prefix and whitespace.
    
    Args:
        response: Raw LLM response
    
    Returns:
        str: Cleaned contextual description
    """
    response_str = str(response)
    if "assistant: " in response_str:
        cleaned = response_str.split("assistant: ")[1].strip()
    else:
        cleaned = response_str.strip()
    return cleaned


def process_documents_batch(documents, node_parser, gen_model, batch_size=10, max_context_tokens=1500):
    """
    Process documents in batches to manage memory and API limits.
    
    Args:
        documents: List of documents to process
        node_parser: LlamaIndex node parser
        gen_model: Ollama model instance for generating context
        batch_size: Number of chunks to process in each batch
        max_context_tokens: Maximum tokens for surrounding context
    
    Returns:
        dict: Contains all updated nodes with context in metadata, organized by document
    """
    all_updated_nodes = {}
    
    for doc_idx, document in enumerate(documents):
        print(f"Processing document {doc_idx + 1}/{len(documents)}")
        
        # Get nodes for this document
        nodes = node_parser.get_nodes_from_documents([document])
        print(f"Document {doc_idx+1} has {len(nodes)} chunks")
        
        updated_nodes = [None] * len(nodes)
        
        # Process in batches
        for i in range(0, len(nodes), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(nodes))))
            print(f"  Processing batch {i//batch_size + 1} (chunks {i}-{min(i + batch_size - 1, len(nodes) - 1)})")
            
            for chunk_idx in batch_indices:
                try:
                    current_chunk = nodes[chunk_idx]
                    
                    # Get surrounding pages context
                    context_info = get_surrounding_pages_context(
                        [document], nodes, chunk_idx, max_context_tokens
                    )
                    
                    # Create prompt
                    prompt = create_contextual_prompt(current_chunk.text, context_info["context"])
                    
                    # Generate context with LLM using ChatMessage format
                    messages = [
                        ChatMessage(role="system", content="You are helpful AI Assistant."),
                        ChatMessage(role="user", content=prompt),
                    ]
                    raw_contextual_description = gen_model.chat(messages)
                    
                    # Clean the LLM response and add to metadata
                    cleaned_context = clean_llm_response(raw_contextual_description)
                    current_chunk.metadata["context"] = cleaned_context
                    
                    updated_nodes[chunk_idx] = current_chunk
                    
                except Exception as e:
                    print(f"    Error processing chunk {chunk_idx}: {e}")
                    updated_nodes[chunk_idx] = nodes[chunk_idx]
                    continue
        
        # Fill any remaining None values with original nodes
        for idx, node in enumerate(updated_nodes):
            if node is None:
                updated_nodes[idx] = nodes[idx]
        
        all_updated_nodes[f"document_{doc_idx}"] = updated_nodes
        print(f"Completed document {doc_idx + 1}")
    
    return all_updated_nodes

# ============================================================================
# FILE UTILITIES
# ============================================================================

def clean_filename(filename):
    """
    Clean filename for use in file paths by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        str: Cleaned filename safe for file system
    """
    # Remove file extension and clean up
    name_without_ext = os.path.splitext(filename)[0]
    
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^\w\-_\.]', '_', name_without_ext)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned


def get_document_filename(nodes):
    """
    Extract filename from node metadata.
    
    Args:
        nodes: List of nodes
    
    Returns:
        str: Cleaned filename or fallback name
    """
    for node in nodes:
        if "file_name" in node.metadata:
            return clean_filename(node.metadata["file_name"])
    
    # Fallback if no filename found
    return "unknown_document"


def save_nodes_separately(updated_nodes_dict, output_dir="contextualized_nodes"):
    """
    Save each document's nodes as a separate JSON file.
    
    Args:
        updated_nodes_dict: Dictionary from process_documents_* functions
        output_dir: Directory to save the files
    
    Returns:
        list: List of paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    for doc_key, nodes in updated_nodes_dict.items():
        # Get actual filename for this document
        doc_filename = get_document_filename(nodes)
        
        nodes_data = []
        for i, node in enumerate(nodes):
            try:
                node_dict = node.to_dict()
                nodes_data.append(node_dict)
            except Exception as e:
                print(f"Error serializing node {i} from {doc_key}: {e}")
                continue
        
        # Create document data structure
        document_data = {
            "document_filename": doc_filename,
            "original_file": nodes[0].metadata.get("file_name", "unknown") if nodes else "unknown",
            "file_path": nodes[0].metadata.get("file_path", "unknown") if nodes else "unknown",
            "processing_metadata": {
                "created_at": datetime.now().isoformat(),
                "node_count": len(nodes_data),
                "doc_key": doc_key
            },
            "nodes": nodes_data
        }
        
        # Create filename for this document
        safe_filename = f"{doc_filename}_contextualized.json"
        filepath = os.path.join(output_dir, safe_filename.lower())
        
        # Save the document
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filepath)
        print(f"Saved {len(nodes_data)} nodes for '{doc_filename}' to {filepath}")
    
    print(f"✅ Saved {len(saved_files)} documents separately")
    return saved_files


def load_all_nodes_from_directory(nodes_dir="contextualized_nodes"):
    """
    Load all nodes from all JSON files in a directory.
    
    Args:
        nodes_dir: Directory containing the JSON files
    
    Returns:
        tuple: (dict of loaded nodes organized by document, flat list of all nodes)
    """
    if not os.path.exists(nodes_dir):
        raise FileNotFoundError(f"Directory not found: {nodes_dir}")
    
    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(nodes_dir) if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {nodes_dir}")
    
    print(f"Found {len(json_files)} JSON files in {nodes_dir}")
    
    loaded_nodes = {}
    all_nodes = []  # Flat list for indexing
    total_nodes_loaded = 0
    
    for json_file in json_files:
        filepath = os.path.join(nodes_dir, json_file)
        print(f"Loading: {json_file}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract document information
            doc_filename = data.get("document_filename", os.path.splitext(json_file)[0])
            original_file = data.get("original_file", "unknown")
            node_count = data.get("processing_metadata", {}).get("node_count", 0)
            
            # Load nodes
            nodes = []
            for node_dict in data["nodes"]:
                try:
                    node = TextNode.from_dict(node_dict)
                    nodes.append(node)
                    all_nodes.append(node)
                except Exception as e:
                    print(f"  Error loading node from {json_file}: {e}")
                    continue
            
            loaded_nodes[doc_filename] = nodes
            total_nodes_loaded += len(nodes)
            print(f"  ✅ Loaded {len(nodes)} nodes for '{original_file}'")
            
        except Exception as e:
            print(f"  ❌ Error loading file {json_file}: {e}")
            continue
    
    print(f"✅ Successfully loaded {total_nodes_loaded} total nodes from {len(loaded_nodes)} documents")
    return loaded_nodes, all_nodes


def list_saved_documents(nodes_dir="contextualized_nodes"):
    """
    List all saved document files in the directory with their metadata.
    
    Args:
        nodes_dir: Directory containing the JSON files
    
    Returns:
        list: List of document information dictionaries
    """
    if not os.path.exists(nodes_dir):
        print(f"Directory not found: {nodes_dir}")
        return []
    
    json_files = [f for f in os.listdir(nodes_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in directory: {nodes_dir}")
        return []
    
    documents_info = []
    
    print(f"Found {len(json_files)} saved documents in {nodes_dir}:")
    print("-" * 80)
    
    for json_file in sorted(json_files):
        filepath = os.path.join(nodes_dir, json_file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_info = {
                "filename": json_file,
                "document_name": data.get("document_filename", "unknown"),
                "original_file": data.get("original_file", "unknown"),
                "node_count": data.get("processing_metadata", {}).get("node_count", 0),
                "created_at": data.get("processing_metadata", {}).get("created_at", "unknown"),
                "file_path": filepath
            }
            
            documents_info.append(doc_info)
            
            print(f"📄 {doc_info['document_name']}")
            print(f"   Original: {doc_info['original_file']}")
            print(f"   Nodes: {doc_info['node_count']}")
            print(f"   Created: {doc_info['created_at']}")
            print(f"   File: {json_file}")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
            print()
    
    print(f"Total: {len(documents_info)} documents with {sum(doc['node_count'] for doc in documents_info)} nodes")
    return documents_info

# ============================================================================
# DATABASE SETUP FUNCTIONS
# ============================================================================

def setup_database(connection_string, db_name):
    """
    Set up PostgreSQL database for vector storage.
    
    Args:
        connection_string: PostgreSQL connection string
        db_name: Name of the database to create
    """
    print(f"Setting up database: {db_name}")
    
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
    
    conn.close()
    print(f"Database {db_name} created successfully")


def create_vector_store(connection_string, db_name):
    """
    Create and configure the vector store.
    
    Args:
        connection_string: PostgreSQL connection string
        db_name: Name of the database
    
    Returns:
        PGVectorStore: Configured vector store
    """
    print("Creating vector store...")
    
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
    
    print("Vector store created successfully")
    return vector_store

# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def process_and_save_documents(docs_dir=DOCS_DIR, output_dir=OUTPUT_DIR, 
                              batch_size=BATCH_SIZE, max_context_tokens=MAX_CONTEXT_TOKENS):
    """
    PHASE 1: Process documents from directory and save contextualized nodes separately.
    
    Args:
        docs_dir: Directory containing documents to process
        output_dir: Directory to save processed nodes
        batch_size: Number of chunks to process in each batch
        max_context_tokens: Maximum tokens for surrounding context
    
    Returns:
        list: List of paths to saved node files
    """
    print("="*80)
    print("PHASE 1: PROCESSING AND SAVING DOCUMENTS")
    print("="*80)
    
    # Initialize document reader and node parser
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    node_parser = DoclingNodeParser()
    
    # Load documents
    print(f"Loading documents from: {docs_dir}")
    documents = SimpleDirectoryReader(
        input_dir=docs_dir,
        file_extractor={".pdf": reader},
    ).load_data(show_progress=True)
    
    print(f"Loaded {len(documents)} documents")
    
    # Process documents with contextual information
    print("Processing documents with contextual information...")
    updated_nodes_dict = process_documents_batch(
        documents, 
        node_parser, 
        GEN_MODEL, 
        batch_size=batch_size,
        max_context_tokens=max_context_tokens
    )
    
    # Save processed nodes separately
    print("Saving processed nodes separately...")
    saved_file_paths = save_nodes_separately(updated_nodes_dict, output_dir)
    
    print(f"✅ Phase 1 completed! {len(saved_file_paths)} files saved to: {output_dir}")
    return saved_file_paths


def load_and_index_nodes(nodes_dir=OUTPUT_DIR, connection_string=CONNECTION_STRING, db_name=DB_NAME):
    """
    PHASE 2: Load saved nodes from directory and create vector index.
    
    Args:
        nodes_dir: Directory containing the saved nodes JSON files
        connection_string: PostgreSQL connection string
        db_name: Name of the database
    
    Returns:
        VectorStoreIndex: Created vector index
    """
    print("="*80)
    print("PHASE 2: LOADING NODES AND CREATING INDEX")
    print("="*80)
    
    # Load processed nodes from directory
    print(f"Loading nodes from directory: {nodes_dir}")
    loaded_nodes_dict, all_nodes = load_all_nodes_from_directory(nodes_dir)
    
    print(f"Loaded {len(all_nodes)} total nodes from {len(loaded_nodes_dict)} documents")
    
    # Setup database
    setup_database(connection_string, db_name)
    
    # Create vector store
    vector_store = create_vector_store(connection_string, db_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create vector index
    print("Creating vector index...")
    hybrid_index = VectorStoreIndex.from_documents(
        documents=all_nodes,  # Use the flat list of all nodes
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
        show_progress=True
    )
    
    print("✅ Phase 2 completed! Vector index created successfully")
    return hybrid_index


def run_full_pipeline(docs_dir=DOCS_DIR, output_dir=OUTPUT_DIR):
    """
    Run the complete pipeline: process documents and create index.
    
    Args:
        docs_dir: Directory containing documents to process
        output_dir: Directory to save processed nodes
    
    Returns:
        tuple: (list of saved file paths, vector_index)
    """
    print("🚀 Starting full document processing pipeline...")
    
    # Phase 1: Process and save
    saved_file_paths = process_and_save_documents(docs_dir, output_dir)
    
    # Phase 2: Load and index
    vector_index = load_and_index_nodes(output_dir)
    
    print("🎉 Full pipeline completed successfully!")
    return saved_file_paths, vector_index


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Choose execution mode:
    
    # Option 1: Run full pipeline
    # saved_files, index = run_full_pipeline()
    
    # Option 2: Run only Phase 1 (processing and saving)
    # saved_files = process_and_save_documents()
    
    # Run only Phase 2 (loading and indexing from directory)
    index = load_and_index_nodes("../data/processed_docs")
    
