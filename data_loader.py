import os
import logging
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Lazy singleton embedding model instance
_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    """Lazy initializer for local HuggingFace BAAI/bge-small-en-v1.5 embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model


def embed_texts(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    """Generate 384-dimensional local vector embeddings for input texts."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    return [e.tolist() for e in embeddings]


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Loads PDF using LlamaIndex SimpleDirectoryReader and splits into text chunks.
    Cleans Unicode surrogates to prevent hash encoding errors on mathematical PDF symbols.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    try:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
    except Exception as err:
        logger.error(f"Failed to load PDF at {pdf_path}: {err}")
        return []

    if not documents:
        return []

    # Clean lone UTF-8 surrogate characters (common in math/code PDF symbols)
    for doc in documents:
        content = doc.get_content()
        if content:
            cleaned_content = content.encode("utf-8", "ignore").decode("utf-8")
            doc.set_content(cleaned_content)

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    chunks = [node.get_content().strip() for node in nodes if node.get_content().strip()]
    return chunks
