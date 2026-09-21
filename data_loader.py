import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import hashlib
from typing import List, Dict, Any, Optional
from PIL import Image
import io

try:
    import pymupdf
    _HAS_PYMUPDF = True
except Exception:
    try:
        import fitz as pymupdf
        _HAS_PYMUPDF = True
    except Exception:
        _HAS_PYMUPDF = False

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except Exception:
    _HAS_PYTESSERACT = False

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
    """Create a deterministic pseudo-embedding from a SHA256 hash."""
    h = hashlib.sha256(text.encode("utf-8", "ignore")).digest()
    vals = []
    i = 0
    while len(vals) < dim:
        chunk = hashlib.sha256(h + i.to_bytes(2, "big")).digest()
        for b in chunk:
            if len(vals) >= dim:
                break
            vals.append((b / 255.0) * 2.0 - 1.0)
        i += 1
    return vals[:dim]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate 384-dimensional vector embeddings for input texts."""
    if not texts:
        return []

    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = list(model.embed(texts, batch_size=256))
            return [e.tolist() for e in embeddings]
        except Exception:
            pass

    return [_deterministic_hash_vector(t, dim=384) for t in texts]


def _try_ocr_page_image(pil_img: Image.Image) -> str:
    """Run pytesseract on a PIL Image if available."""
    if not _HAS_PYTESSERACT:
        return ""
    try:
        text = pytesseract.image_to_string(pil_img)
        return text.strip()
    except Exception:
        return ""


def load_and_chunk_pdf_structured(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Load PDF and split into structured chunks with page numbers and OCR tagging."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    chunks_data: List[Dict[str, Any]] = []

    # 1. Primary Engine: PyMuPDF with per-page text & OCR fallback
    if _HAS_PYMUPDF:
        try:
            doc = pymupdf.open(pdf_path)
            for page_index in range(len(doc)):
                page_num = page_index + 1
                page = doc[page_index]
                text = page.get_text() or ""
                ocr_used = False

                # If text is nearly empty, try OCR on this page
                if len(text.strip()) < 40 and _HAS_PYTESSERACT:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = _try_ocr_page_image(pil_img)
                        if len(ocr_text.strip()) > len(text.strip()):
                            text = ocr_text
                            ocr_used = True
                    except Exception:
                        pass

                # Clean UTF-8 surrogates
                clean_text = text.encode("utf-8", "ignore").decode("utf-8").strip()
                if not clean_text:
                    continue

                # Sliding window chunking per page
                start = 0
                while start < len(clean_text):
                    end = start + chunk_size
                    chunk = clean_text[start:end].strip()
                    if chunk:
                        chunks_data.append({
                            "text": chunk,
                            "page_number": page_num,
                            "ocr_used": ocr_used,
                        })
                    start = max(0, end - chunk_overlap)
                    if start >= len(clean_text) or end >= len(clean_text):
                        break

            doc.close()
            if chunks_data:
                return chunks_data
        except Exception:
            pass

    # 2. PyPDF fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page_index, page in enumerate(reader.pages):
            page_num = page_index + 1
            txt = (page.extract_text() or "").encode("utf-8", "ignore").decode("utf-8").strip()
            if not txt:
                continue

            start = 0
            while start < len(txt):
                end = start + chunk_size
                chunk = txt[start:end].strip()
                if chunk:
                    chunks_data.append({
                        "text": chunk,
                        "page_number": page_num,
                        "ocr_used": False,
                    })
                start = max(0, end - chunk_overlap)
                if start >= len(txt) or end >= len(txt):
                    break
        return chunks_data
    except Exception:
        pass

    return []


def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """Backwards-compatible wrapper returning plain text chunks."""
    structured = load_and_chunk_pdf_structured(pdf_path, chunk_size, chunk_overlap)
    return [c["text"] for c in structured]
