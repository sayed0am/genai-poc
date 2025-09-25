from sqlalchemy import make_url

import psycopg2
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.text_splitter import TokenTextSplitter

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext

from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.schema import Document, TextNode
from typing import List
import copy
# ============================================================================
# CONFIGURATION
# ============================================================================
# Database settings
CONNECTION_STRING = "postgresql://postgres:password@localhost:5432"
DB_NAME = "vector_db"

# Document processing settings
MDS_DIR = "../data/processed_docs/md"

# model settings
# GEN_MODEL = Ollama(model="qwen3:4b-instruct-2507-q8_0", temperature=0.7, keep_alive=True, context_window=2048)
GEN_MODEL = OpenRouter(
    max_tokens=4096,
    context_window=500000,
    model="openai/gpt-4.1-nano",
)

EMBED_MODEL = OllamaEmbedding(model_name="embeddinggemma:300m-qat-q8_0")

# ============================================================================
# Vector Store Setup
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


def create_contextual_nodes(nodes_, whole_document):
    """Function to create contextual nodes for a list of nodes"""
    nodes_modified = []
    prompt_chunk = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {CHUNK_CONTENT}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else
    """

    for idx, node in enumerate(nodes_):
        new_node = copy.deepcopy(node)
        
        # Cache the document in system message - this gets cached once and reused
        messages = [
            ChatMessage(
                role="system", 
                content=[
                    TextBlock(text="You are a helpful AI Assistant."),
                    TextBlock(
                        text=f"<document>\n{whole_document}\n</document>",
                    )
                ]
            ),
            # Only the chunk-specific content goes in user message (not cached)
            ChatMessage(
                role="user",
                content=[
                    TextBlock(
                        text=prompt_chunk.format(CHUNK_CONTENT=node.text)
                    )
                ]
            )
        ]
        print(f"Generating context for chunk {idx+1}/{len(nodes_)}...")
        new_node.metadata["context"] = str(
            GEN_MODEL.chat(
                messages
            )
        ).split("assistant:")[-1].strip()
        nodes_modified.append(new_node)
    
    return nodes_modified


def read_markdown_files(md_dir):
    """
    Read markdown files from a directory and return documents.

    Args:
        md_dir: Directory containing markdown files
    """
    print(f"Reading markdown files from {md_dir}...")
    reader = SimpleDirectoryReader(input_dir=md_dir, required_exts=[".md"])
    documents = reader.load_data()

    # Remove <!-- image --> from all documents
    cleaned_documents = []
    for doc in documents:
        cleaned_text = doc.text.replace("<!-- image -->\n\n", "")
        cleaned_doc = Document(text=cleaned_text, metadata=doc.metadata)
        cleaned_documents.append(cleaned_doc)

    print(f"Loaded {len(cleaned_documents)} documents")

    return cleaned_documents


def create_page_aware_nodes(markdown_text: str, filename: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[TextNode]:
    """Split markdown by pages, then by tokens, keeping page numbers and filename"""
    
    # Step 1: Split by page breaks
    pages = markdown_text.split('<!-- page break -->')

    # Step 2: Setup token splitter
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")
    
    all_nodes = []
    
    # Step 3: Process each page
    for page_num, page_content in enumerate(pages, 1):
        page_content = page_content.strip()
        if not page_content:
            continue
            
        # Split page into token chunks
        chunks = splitter.split_text(page_content)
        
        # Create nodes with page and filename metadata
        for chunk in chunks:
            node = TextNode(
                text=chunk,
                metadata={
                    "filename": filename,
                    "page_number": page_num
                }
            )
            
            all_nodes.append(node)
    
    return all_nodes



if __name__ == "__main__":
    print("Setting up database...")
    # setup_database(CONNECTION_STRING, DB_NAME)
    vector_store = create_vector_store(CONNECTION_STRING, DB_NAME)
    print("Database and vector store setup complete.")

    print("Reading markdown files...")
    docs = read_markdown_files(MDS_DIR)

    print(f"Found {len(docs)} markdown files")
    
    all_nodes = []
    
    for doc in docs[2:]:
        print(f"Processing {doc.metadata['file_name']}...")
        
        # Create page-aware nodes
        nodes = create_page_aware_nodes(
            doc.text, 
            doc.metadata["file_name"]
        )
        print(f"Created {len(nodes)} nodes from {doc.metadata['file_name']}")
        print("Adding contextual information...")
        contextual_nodes = create_contextual_nodes(nodes, doc.text)
        print(f"Created {len(contextual_nodes)} contextual nodes")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        hybrid_index = VectorStoreIndex.from_documents(
            documents=contextual_nodes,  # Use the flat list of all nodes
            storage_context=storage_context,
            embed_model=EMBED_MODEL,
            show_progress=True
        )
        print(f"Added {len(contextual_nodes)} nodes to vector store")

