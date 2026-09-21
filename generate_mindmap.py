import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ── Color Palette (Modern, Professional Tech Theme) ──────────────────────────
BG_COLOR = "#0F172A"       # Slate 900
CARD_BG = "#1E293B"        # Slate 800
BORDER_COLOR = "#334155"   # Slate 700
TEXT_MAIN = "#F8FAFC"      # Slate 50
TEXT_MUTED = "#94A3B8"     # Slate 400

BRANCH_COLORS = {
    "center": ("#6366F1", "#818CF8", "#312E81"),   # Indigo
    "api": ("#06B6D4", "#38BDF8", "#083344"),      # Cyan / Sky
    "ingest": ("#10B981", "#34D399", "#064E3B"),   # Emerald / Green
    "embed": ("#F59E0B", "#FBBF24", "#451A03"),    # Amber / Orange
    "vector": ("#8B5CF6", "#A78BFA", "#2E1065"),   # Violet / Purple
    "llm": ("#EC4899", "#F472B6", "#500724"),      # Pink / Rose
    "inngest": ("#3B82F6", "#60A5FA", "#172554"),  # Blue
}

def create_mindmap_image(output_png_path: str):
    fig = plt.figure(figsize=(24, 14), dpi=300, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 1400)
    ax.axis('off')

    # Draw Title Header
    ax.text(1200, 1340, "RAG INSTITUTE ASSISTANT — BACKEND ARCHITECTURE MIND MAP",
            ha="center", va="center", fontsize=22, fontweight="bold", color="#FFFFFF", family="sans-serif")
    ax.text(1200, 1305, "FastAPI • LlamaIndex & PyPDF • FastEmbed • Qdrant Vector DB • Inngest Orchestration • OpenRouter LLM",
            ha="center", va="center", fontsize=12, color="#94A3B8", family="sans-serif")

    # Helper function to draw rounded cards
    def draw_node(x, y, w, h, title, items, color_key, title_size=11, item_size=8.5, radius=12):
        primary, border, bg_fill = BRANCH_COLORS[color_key]
        
        # Shadow
        shadow = patches.FancyBboxPatch((x - w/2 + 4, y - h/2 - 4), w, h,
                                        boxstyle=f"round,pad=0,rounding_size={radius}",
                                        facecolor="#050811", edgecolor="none", alpha=0.4, zorder=2)
        ax.add_patch(shadow)

        # Card Box
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle=f"round,pad=0,rounding_size={radius}",
                                     facecolor=CARD_BG, edgecolor=border, linewidth=1.8, zorder=3)
        ax.add_patch(box)

        # Title Header Bar inside Card
        header_h = 32
        header_box = patches.FancyBboxPatch((x - w/2 + 2, y + h/2 - header_h - 2), w - 4, header_h,
                                            boxstyle=f"round,pad=0,rounding_size={radius-2}",
                                            facecolor=bg_fill, edgecolor=border, linewidth=0.8, zorder=4)
        ax.add_patch(header_box)

        # Title Text
        ax.text(x, y + h/2 - header_h/2 - 2, title,
                ha="center", va="center", fontsize=title_size, fontweight="bold",
                color=primary, family="sans-serif", zorder=5)

        # Items Text
        if items:
            start_y = y + h/2 - header_h - 18
            line_spacing = (h - header_h - 25) / max(len(items), 1)
            for i, it in enumerate(items):
                cur_y = start_y - (i * line_spacing)
                # Bullet or text
                ax.text(x - w/2 + 16, cur_y, f"• {it}",
                        ha="left", va="center", fontsize=item_size,
                        color="#E2E8F0", family="sans-serif", zorder=5)

    # Helper function for curved connector lines
    def draw_connector(x1, y1, x2, y2, color, lw=2.5, style="-"):
        mid_x = (x1 + x2) / 2
        path = matplotlib.path.Path(
            [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)],
            [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4]
        )
        patch = patches.PathPatch(path, facecolor="none", edgecolor=color, linewidth=lw, linestyle=style, zorder=1, alpha=0.85)
        ax.add_patch(patch)

    # ── Central Core Node ──
    cx, cy = 1200, 680
    c_w, c_h = 300, 130
    
    # Draw Center Node
    c_shadow = patches.FancyBboxPatch((cx - c_w/2 + 5, cy - c_h/2 - 5), c_w, c_h,
                                      boxstyle="round,pad=0,rounding_size=18",
                                      facecolor="#050811", alpha=0.5, zorder=2)
    ax.add_patch(c_shadow)
    c_box = patches.FancyBboxPatch((cx - c_w/2, cy - c_h/2), c_w, c_h,
                                   boxstyle="round,pad=0,rounding_size=18",
                                   facecolor="#1E1B4B", edgecolor="#818CF8", linewidth=3, zorder=3)
    ax.add_patch(c_box)
    ax.text(cx, cy + 28, "⚡ FASTAPI CORE BACKEND", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#A5B4FC", zorder=5)
    ax.text(cx, cy - 2, "Production RAG Engine Controller", ha="center", va="center",
            fontsize=10, color="#E0E7FF", zorder=5)
    ax.text(cx, cy - 28, "main.py • CORS • Pydantic Models • Inngest", ha="center", va="center",
            fontsize=9, color="#94A3B8", zorder=5)

    # ── BRANCH 1: API Endpoints (Top Left) ──
    b1_x, b1_y = 650, 1080
    draw_connector(cx, cy + c_h/2 - 10, b1_x, b1_y, BRANCH_COLORS["api"][0])
    draw_node(b1_x, b1_y, 350, 190, "1. REST API ENDPOINTS", [
        "POST /ingest — Direct JSON PDF Ingestion",
        "POST /upload — Multipart File Upload (auto-cleanup)",
        "POST /upload-batch — Bulk Multi-PDF Ingestion",
        "POST /ask — Semantic Vector Query & LLM Answer",
        "GET /sources — Active Document IDs List",
        "GET /, /health — Server Liveness & API Discovery",
        "GET /api/inngest — FastApi Inngest Webhook Bridge"
    ], "api", title_size=11, item_size=8.5)

    # Sub-node: Request / Response Types
    sub1_x, sub1_y = 230, 1080
    draw_connector(b1_x - 175, b1_y, sub1_x + 150, sub1_y, BRANCH_COLORS["api"][1], lw=1.5, style=":")
    draw_node(sub1_x, sub1_y, 300, 140, "Pydantic Schemas (custom_types.py)", [
        "IngestPDFData(pdf_path, source_id, replace)",
        "IngestResult(status, source_id, chunks_ingested)",
        "AskRequest(question, top_k, source_id)",
        "AskResponse(answer, sources, num_contexts)"
    ], "api", title_size=9.5, item_size=7.8)

    # ── BRANCH 2: Ingestion & Parsing (Bottom Left) ──
    b2_x, b2_y = 650, 720
    draw_connector(cx - c_w/2 + 10, cy, b2_x + 175, b2_y, BRANCH_COLORS["ingest"][0])
    draw_node(b2_x, b2_y, 350, 180, "2. INGESTION & PARSING (data_loader.py)", [
        "LlamaIndex Engine (Primary)",
        "  • SimpleDirectoryReader document loader",
        "  • SentenceSplitter (chunk: 500, overlap: 50)",
        "  • UTF-8 surrogate sanitization filter",
        "PyPDF Fallback Engine (Resilience)",
        "  • PdfReader page text extraction",
        "  • Dynamic sliding-window text chunker"
    ], "ingest", title_size=11, item_size=8.2)

    # ── BRANCH 3: Embedding Layer (Bottom Left lower) ──
    b3_x, b3_y = 650, 320
    draw_connector(cx, cy - c_h/2 + 10, b3_x, b3_y, BRANCH_COLORS["embed"][0])
    draw_node(b3_x, b3_y, 350, 180, "3. EMBEDDING GENERATION (data_loader.py)", [
        "FastEmbed Framework (BAAI/bge-small-en-v1.5)",
        "  • Local inference (ONNX Runtime accelerated)",
        "  • 384-Dimensional Dense Vector Embeddings",
        "  • High-throughput batch processing (batch: 256)",
        "Deterministic Hash Fallback (No-GPU/Test mode)",
        "  • Multi-round SHA-256 Pseudo-Embedding",
        "  • Normalized floats in [-1.0, 1.0] for CI"
    ], "embed", title_size=11, item_size=8.2)

    # ── BRANCH 4: Vector Storage & Search (Bottom Right) ──
    b4_x, b4_y = 1750, 320
    draw_connector(cx, cy - c_h/2 + 10, b4_x, b4_y, BRANCH_COLORS["vector"][0])
    draw_node(b4_x, b4_y, 360, 200, "4. VECTOR DATABASE (vector_db.py)", [
        "Qdrant Storage (Singleton Pattern)",
        "  • Remote Qdrant Cloud (QDRANT_URL + KEY)",
        "  • Local On-Disk Qdrant (path: ./qdrant_db)",
        "  • In-Memory Dict Fallback for zero-dep tests",
        "Collection: 'rag_documents'",
        "  • Metric: Cosine Distance (dim=384)",
        "  • Payload Schema: Keyword index on 'source_id'",
        "  • Batch Upserts (batch size: 50 points, UUIDv4)",
        "  • Filtered Search & Source ID Scoping"
    ], "vector", title_size=11, item_size=8)

    # ── BRANCH 5: RAG Query & LLM Generation (Middle Right) ──
    b5_x, b5_y = 1750, 720
    draw_connector(cx + c_w/2 - 10, cy, b5_x - 180, b5_y, BRANCH_COLORS["llm"][0])
    draw_node(b5_x, b5_y, 360, 210, "5. RAG & LLM SYNTHESIS (main.py)", [
        "Query Embedding: embed_texts([question])",
        "Semantic Search: Qdrant Cosine Top-K matches",
        "Context Assembly: Formatted [1], [2] citations",
        "Strict Grounding System Prompt:",
        "  • Source-backed factual response requirement",
        "  • Clear distinction of general fallback info",
        "OpenRouter Multi-Model Fallback Chain:",
        "  1. minimax/minimax-m3:free",
        "  2. poolside/laguna-s-2.1:free",
        "  3. nvidia/nemotron-3-nano-omni-30b (reasoning)",
        "  4. nvidia/nemotron-3-super-120b:free",
        "  5. Live OpenRouter Catalog Auto-Discovery"
    ], "llm", title_size=11, item_size=8)

    # ── BRANCH 6: Background Orchestration (Top Right) ──
    b6_x, b6_y = 1750, 1080
    draw_connector(cx, cy + c_h/2 - 10, b6_x, b6_y, BRANCH_COLORS["inngest"][0])
    draw_node(b6_x, b6_y, 360, 190, "6. ASYNC ORCHESTRATION (Inngest)", [
        "Event Trigger: 'rag/ingest_pdf'",
        "Concurrency Control & Protection:",
        "  • Throttle: max 2 executions / 60 seconds",
        "  • Rate Limit: 1 per 4h per source_id",
        "Durable Multi-Step Workflow (step.run):",
        "  • Step 1: 'extract-chunks' (LlamaIndex parse)",
        "  • Step 2: 'embed-chunks' (FastEmbed vector)",
        "  • Step 3: 'store-qdrant' (Qdrant upsert)",
        "Automatic retry, crash recovery & execution history"
    ], "inngest", title_size=11, item_size=8)

    # Sub-node: Environment & Config (Top center)
    sub_env_x, sub_env_y = 1200, 1080
    draw_connector(cx, cy + c_h/2, sub_env_x, sub_env_y - 65, "#64748B", lw=1.5, style="--")
    draw_node(sub_env_x, sub_env_y, 320, 130, "Environment & Configuration (.env)", [
        "OPENROUTER_API_KEY — LLM API auth",
        "QDRANT_URL / KEY / LOCAL_PATH — DB config",
        "INNGEST_DEV_EVENT_URL — Queue URL",
        "FASTAPI_BASE_URL — UI connection",
        "OPENROUTER_MODEL — Custom model override"
    ], "center", title_size=10, item_size=8)

    # Legend / Flow Footer
    legend_box = patches.FancyBboxPatch((400, 35), 1600, 50,
                                        boxstyle="round,pad=0,rounding_size=10",
                                        facecolor="#1E293B", edgecolor="#334155", linewidth=1.2, zorder=3)
    ax.add_patch(legend_box)
    ax.text(1200, 60, "RAG INGESTION FLOW:  PDF Document  ➔  LlamaIndex / PyPDF  ➔  FastEmbed (bge-small 384d)  ➔  Qdrant Storage\n"
                      "RAG QUERY FLOW:  User Question  ➔  Query Vector  ➔  Qdrant Cosine Top-K  ➔  Context Formatter  ➔  OpenRouter Free LLMs",
            ha="center", va="center", fontsize=9, fontweight="semibold", color="#38BDF8", family="sans-serif", zorder=5)

    plt.savefig(output_png_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Mind map PNG saved to {output_png_path}")


def create_mindmap_pdf(png_path: str, output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Page 1: High-Res Diagram Showcase
    story.append(Paragraph("RAG Institute Assistant — Backend Architecture Mind Map", title_style))
    story.append(Paragraph("Visual hierarchy of FastAPI services, document parsing, embeddings, vector indexing, Inngest orchestration, and LLM synthesis.", subtitle_style))
    
    # 11 x 8.5 letter landscape in points is 792 x 612. Usable width: 792 - 72 = 720. Usable height: 612 - 72 = 540.
    # Image aspect ratio 24:14 = 1.714. At 700 width, height = 408.
    story.append(RLImage(png_path, width=720, height=420))
    story.append(PageBreak())

    # Page 2: Detailed Technical Architecture Specification
    story.append(Paragraph("Backend Architectural Components & Data Flow", title_style))
    story.append(Paragraph("Comprehensive breakdown of backend modules, endpoints, resilient fallbacks, and execution pipelines.", subtitle_style))

    table_data = [
        [
            Paragraph("<b>Component / Layer</b>", body_style),
            Paragraph("<b>Implementation File</b>", body_style),
            Paragraph("<b>Primary Technologies</b>", body_style),
            Paragraph("<b>Key Responsibilities & Fallbacks</b>", body_style)
        ],
        [
            Paragraph("<b>1. API Controller</b>", body_style),
            Paragraph("<font name='Courier'>main.py</font>", code_style),
            Paragraph("FastAPI, Uvicorn, Pydantic, CORS", body_style),
            Paragraph("Exposes REST endpoints (<font name='Courier'>/ingest, /upload, /upload-batch, /ask, /sources, /health</font>). Handles input validation and orchestrates pipelines.", body_style)
        ],
        [
            Paragraph("<b>2. Parsing & Chunking</b>", body_style),
            Paragraph("<font name='Courier'>data_loader.py</font>", code_style),
            Paragraph("LlamaIndex (SentenceSplitter)<br/>PyPDF (Fallback)", body_style),
            Paragraph("Extracts text from PDF documents. Chunks text into 500-token windows with 50-token overlap. Sanitizes UTF-8 surrogate characters. Falls back to PyPDF sliding window.", body_style)
        ],
        [
            Paragraph("<b>3. Vector Embeddings</b>", body_style),
            Paragraph("<font name='Courier'>data_loader.py</font>", code_style),
            Paragraph("FastEmbed (bge-small-en-v1.5)<br/>Deterministic SHA-256 (Fallback)", body_style),
            Paragraph("Generates 384-dimensional dense vectors using local ONNX inference. When offline or in CI test mode, computes deterministic SHA-256 float vectors without ML models.", body_style)
        ],
        [
            Paragraph("<b>4. Vector Storage</b>", body_style),
            Paragraph("<font name='Courier'>vector_db.py</font>", code_style),
            Paragraph("Qdrant Client (Cloud / Disk / In-Memory)", body_style),
            Paragraph("Singleton storage wrapper. Manages <font name='Courier'>rag_documents</font> collection with Cosine similarity. Supports batch upserts (50/batch), keyword indexing on <font name='Courier'>source_id</font>, and metadata filtering.", body_style)
        ],
        [
            Paragraph("<b>5. Async Orchestration</b>", body_style),
            Paragraph("<font name='Courier'>main.py</font>", code_style),
            Paragraph("Inngest SDK (Event-driven)", body_style),
            Paragraph("Orchestrates background document ingestion with durability. Configured with 2-per-minute throttle and 1-per-4h rate limit per source. 3-step pipeline: chunking ➔ embedding ➔ storage.", body_style)
        ],
        [
            Paragraph("<b>6. LLM Synthesis</b>", body_style),
            Paragraph("<font name='Courier'>main.py</font>", code_style),
            Paragraph("OpenRouter API, OpenAI Client", body_style),
            Paragraph("Constructs context blocks with [1], [2] citations. Executes resilient fallback chain across free models (Gemma-4 31B, Nemotron Nano 9B, Gemma-4 26B, GPT-OSS 20B).", body_style)
        ],
    ]

    t = Table(table_data, colWidths=[110, 95, 160, 355])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 14))

    # Pipeline summary block
    story.append(Paragraph("<b>End-to-End Data Lifecycle</b>", h2_style))
    story.append(Paragraph(
        "<b>• Ingestion Pipeline:</b> <font name='Courier'>PDF Upload ➔ Temp Storage ➔ LlamaIndex/PyPDF Token Chunker ➔ FastEmbed 384d Generator ➔ Qdrant Batch Upsert ➔ Temp Cleanup</font><br/>"
        "<b>• Query Pipeline:</b> <font name='Courier'>User Question ➔ Query FastEmbed Vector ➔ Qdrant Cosine Similarity Search ➔ Context Assembly with Citations ➔ OpenRouter LLM Call ➔ Grounded Answer</font>",
        body_style
    ))

    doc.build(story)
    print(f"Mind map PDF saved to {output_pdf_path}")


if __name__ == "__main__":
    png_out = "/Users/apurva/CSPIT/Sem-5/SGP/backend_mindmap.png"
    pdf_out = "/Users/apurva/CSPIT/Sem-5/SGP/backend_mindmap.pdf"
    create_mindmap_image(png_out)
    create_mindmap_pdf(png_out, pdf_out)
