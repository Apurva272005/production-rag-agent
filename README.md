# RAG Institute Assistant

Production-ready RAG (Retrieval-Augmented Generation) document assistant using FastAPI, Inngest, LlamaIndex, FastEmbed, Qdrant, OpenRouter (Llama 3.1 8B), and Streamlit.

## Setup & Running

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Start FastAPI Backend**:
   ```bash
   uv run python main.py
   ```

3. **Start Inngest Dev Server**:
   ```bash
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
   ```

4. **Start Streamlit UI**:
   ```bash
   uv run streamlit run streamlit_app.py
   ```
