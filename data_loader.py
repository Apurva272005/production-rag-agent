import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import hashlib
from typing import List, Optional

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    _HAS_LLAMA_INDEX = True
except Exception:
    _HAS_LLAMA_INDEX = False

try:
    from fastembed import TextEmbedding
    _HAS_FASTEMBED = True
except Exception:
    _HAS_FASTEMBED = False

# Lazy singleton embedding model instance (may be a fastembed model or None)
_embedding_model: Optional[object] = None


def get_embedding_model() -> Optional[object]:
    """Lazy initializer for local embedding model. Falls back to None on failure."""
    global _embedding_model
    if _embedding_model is None and _HAS_FASTEMBED:
        try:
            _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            _embedding_model = None
    return _embedding_model


def _deterministic_hash_vector(text: str, dim: int = 384) -> List[float]:
    """Create a deterministic pseudo-embedding from a SHA256 hash.

    Produces floats in range [-1,1]. Used as a fallback for tests and
    environments without the heavy embedding dependency.
    """
    h = hashlib.sha256(text.encode("utf-8", "ignore")).digest()
    vals = []
    i = 0
    while len(vals) < dim:
        # Expand by hashing with counter
        chunk = hashlib.sha256(h + i.to_bytes(2, "big")).digest()
        for b in chunk:
            if len(vals) >= dim:
                break
            vals.append((b / 255.0) * 2.0 - 1.0)
        i += 1
    return vals[:dim]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate 384-dimensional vector embeddings for input texts.

    Uses `fastembed` when available; otherwise falls back to a deterministic
    hashed-vector implementation so unit tests and CI can run without
    downloading models.
    """
    if not texts:
        return []

    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = list(model.embed(texts, batch_size=256))
            return [e.tolist() for e in embeddings]
        except Exception:
            pass

    # Fallback deterministic vectors
    return [_deterministic_hash_vector(t, dim=384) for t in texts]


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Load PDF and split into text chunks.

    Prefer `llama_index` when available; otherwise use a simple extraction
    via `pypdf` and a character-based chunking strategy.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    if _HAS_LLAMA_INDEX:
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        if not documents:
            return []
        
        # Clean UTF-8 surrogates to prevent sha256 hash encoding errors
        for doc in documents:
            content = doc.get_content()
            if content:
                clean_content = content.encode("utf-8", "ignore").decode("utf-8")
                doc.set_content(clean_content)

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        chunks = [node.get_content().strip() for node in nodes if node.get_content().strip()]
        return chunks

    # Lightweight fallback: extract text using pypdf and simple sliding-window
    try:
        from pypdf import PdfReader
    except Exception:
        raise RuntimeError("No PDF backend available (install 'llama-index' or 'pypdf').")

    reader = PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        full_text.append(txt)

    text = "\n".join(full_text).strip()
    if not text:
        return []

    # Clean UTF-8 surrogates
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Simple chunking by characters with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - chunk_overlap)
    return chunks
