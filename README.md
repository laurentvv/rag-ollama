# 🧠 RAG-Ollama: Advanced Local RAG with AI-Powered OCR

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Integrated-orange)
![LangChain](https://img.shields.io/badge/LangChain-Powered-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**RAG-Ollama** is a cutting-edge Retrieval-Augmented Generation (RAG) system designed for **local deployment**. It leverages the power of **Ollama** for LLMs and Embeddings, combined with a **Hybrid Search** mechanism (Vector + BM25) and **Smart Document Processing** that uses Vision Language Models (VLM) for high-fidelity OCR.

## 🚀 Key Features

*   **🤖 Fully Local**: Runs entirely on your machine using Ollama and local embeddings.
*   **👁️ Smart AI OCR**: Automatically detects images in documents and uses **Vision Language Models** (via `pdf-ocr-ai`) to describe screenshots and diagrams.
*   **🔍 Hybrid Search**: Combines semantic vector search (ChromaDB) with keyword-based search (BM25) for superior retrieval accuracy.
*   **📄 Multi-Format Support**: Handles PDF, DOCX, PPTX, and Markdown files with intelligent routing.
*   **✅ Automated Verification**: Includes a built-in test suite to ensure system reliability.

## 🛠️ Architecture

```mermaid
graph TD
    subgraph "Document Ingestion"
        A[Source Docs] --> B{File Type?}
        B -- PDF --> C[pdf-ocr-ai (VLM)]
        B -- DOCX/PPTX --> D{Has Images?}
        D -- Yes --> E[Docling (Full OCR)]
        D -- No --> F[Docling (Fast)]
        C --> G[Markdown]
        E --> G
        F --> G
    end

    subgraph "RAG Pipeline"
        G --> H[Chunking]
        H --> I[Vector Store (Chroma)]
        H --> J[BM25 Index]
        K[User Query] --> L{Hybrid Retriever}
        I --> L
        J --> L
        L --> M[Context]
        M --> N[LLM (Ollama)]
        N --> O[Answer]
    end
```

## 📋 Prerequisites

*   **Python 3.13+**
*   **[Ollama](https://ollama.com/)**: Running locally.
*   **[Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)**: Required for Docling.
*   **[LM Studio](https://lmstudio.ai/)** (Optional): For `pdf-ocr-ai` vision capabilities.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/RAG-Ollama.git
    cd RAG-Ollama
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull Ollama Models**:
    ```bash
    ollama pull gemma3:12b
    ollama pull embeddinggemma:latest
    ```

## 🚦 Usage

### 1. Prepare Documents
Place your documents in `C:\test\docs_test` (or configure `SOURCE_DIR` in `prepare_documents.py`).

```bash
python prepare_documents.py
```
*This script will intelligently process your files, using AI OCR for PDFs and fast conversion for text-heavy documents.*

### 2. Run RAG System
Start the interactive RAG CLI:

```bash
python rag_ollama_new.py
```

### 3. Run Verification Tests
Validate the system's performance:

```bash
python test_rag_automated.py
```

## 📂 Project Structure

*   `rag_ollama_new.py`: **Main RAG implementation** with Hybrid Search.
*   `prepare_documents.py`: **Smart document ingestion** and OCR pipeline.
*   `test_rag_automated.py`: **Automated verification suite**.
*   `processed_md/`: Directory containing the generated Markdown files.
*   `chroma_db/`: Local vector database storage.
*   `simple_ollama_cli.py`: *Deprecated legacy script (no longer used).*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.