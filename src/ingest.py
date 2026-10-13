"""
Ingestion pipeline: documents → chunks → embeddings → pgvector.

Run: python -m src.ingest
"""
import sys
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from src import config


def build_vector_store() -> PGVectorStore:
    """Connect to pgvector and return a LlamaIndex-compatible store."""
    return PGVectorStore.from_params(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        database=config.PG_DATABASE,
        table_name=config.TABLE_NAME,
        embed_dim=config.EMBED_DIM,
        # HNSW index config — these are the knobs we discussed
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def build_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def load_documents(docs_dir: str = "data/docs") -> list:
    """Load all supported files from a directory."""
    path = Path(docs_dir)
    if not path.exists() or not any(path.iterdir()):
        print(f"No documents found in {docs_dir}. Add PDFs, .md, or .txt files.")
        sys.exit(1)

    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".html"],
    )
    return reader.load_data()


def ingest():
    print(f"Loading documents from data/docs...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} document(s)")

    print(f"Connecting to pgvector at {config.PG_HOST}:{config.PG_PORT}...")
    vector_store = build_vector_store()

    print(f"Building embed model: {config.EMBED_MODEL}...")
    embed_model = build_embed_model()

    print(f"Chunking: size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    print("Embedding and storing...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"Done. Chunks stored in table '{config.TABLE_NAME}'.")


if __name__ == "__main__":
    ingest()