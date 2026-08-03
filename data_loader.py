import os
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from fastembed import TextEmbedding

# Lazy singleton embedding model instance
_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    """Lazy initializer for local HuggingFace BAAI/bge-small-en-v1.5 embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate 384-dimensional local vector embeddings for input texts."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Loads PDF using LlamaIndex SimpleDirectoryReader and splits into text chunks
    using SentenceSplitter.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    reader = SimpleDirectoryReader(input_files=[pdf_path])
    documents = reader.load_data()

    if not documents:
        return []

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    chunks = [node.get_content().strip() for node in nodes if node.get_content().strip()]
    return chunks
