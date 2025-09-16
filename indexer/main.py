from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

EMBED_MODEL = OllamaEmbedding(model_name="nomic-embed-text")
GEN_MODEL = Ollama(model="qwen3:4b-instruct-2507-q8_0", temperature=0.7, keep_alive=True)
SOURCE = "docs/Abu Dhabi Procurement Standards.PDF" #"https://arxiv.org/pdf/2408.09869"  # Docling Technical Report
QUERY = """
How do the "Delivery Terms" and "Payment Terms" relate to a "Purchase Order" within the procurement process described in this document?

"""

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader

from llama_index.node_parser.docling import DoclingNodeParser

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
node_parser = DoclingNodeParser()

index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
print([(n.text, n.metadata) for n in result.source_nodes])

# reader = DoclingReader()
# node_parser = MarkdownNodeParser()

# # from llama_index.core import SimpleDirectoryReader

# # dir_reader = SimpleDirectoryReader(
# #     input_dir="docs",
# #     file_extractor={".pdf": reader},
# # )

# index = VectorStoreIndex.from_documents(
#     documents=reader.load_data(SOURCE),
#     transformations=[node_parser],
#     embed_model=EMBED_MODEL,
# )
# result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
# print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
# print([(n.text, n.metadata) for n in result.source_nodes])
