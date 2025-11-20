# 🧠 RAG-Ollama

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Integrated-orange)
![LangChain](https://img.shields.io/badge/LangChain-Powered-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**Advanced Local RAG system with AI-Powered OCR and Hybrid Search.**
*Powered by Ollama, ChromaDB, and Vision Language Models.*

## ✨ Features

- **🤖 Fully Local**: Runs entirely on your machine using Ollama and local embeddings.
- **👁️ Smart AI OCR**: Automatically detects images in documents and uses **Vision Language Models** (via `pdf-ocr-ai` logic) to describe screenshots and diagrams.
- **🖼️ Image Support**: Directly processes image files (`.jpg`, `.png`, etc.) using local VLMs.
- **🔍 Hybrid Search**: Combines semantic vector search (ChromaDB) with keyword-based search (BM25) for superior retrieval accuracy.
- **📄 Multi-Format Support**: Handles PDF, DOCX, PPTX, Images, and Markdown files with intelligent routing.
- **⚡ Modern Tooling**: Uses `uv` for fast dependency management and script execution.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

*   **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer and resolver.
*   **[Ollama](https://ollama.com/)**: Running locally for LLM and Embeddings.
*   **[LM Studio](https://lmstudio.ai/)**: Required for AI OCR/Image processing (Vision models).

## 🚀 Workflow (Quick Start)

You don't need to install anything! Just ensure you have the **Prerequisites** ready, then follow these steps using `uvx`.

### 1️⃣ Step 1: Prepare Documents
Convert your source documents (PDF, Images, DOCX) into a clean Markdown format.

```bash
# Process all documents in the source directory
uvx --from git+https://github.com/laurentvv/rag-ollama rag-prepare --input "C:\MyDocs" --output "./processed_md"
```

### 2️⃣ Step 2: Chat & Index
Start the chat interface. It will automatically index new files and let you ask questions.

```bash
uvx --from git+https://github.com/laurentvv/rag-ollama rag-ollama --input "./processed_md" --db "./chroma_db"
```

### ⚡ Quick Add (Single File)
To add a single document without reprocessing everything:

```bash
uvx --from git+https://github.com/laurentvv/rag-ollama rag-add "C:\MyDocs\new_file.pdf" --source-dir "C:\MyDocs" --processed-dir "./processed_md" --db "./chroma_db"
```

---

## 📖 Command Reference

### `rag-prepare` (Document Preparation)
Converts documents to Markdown.
*   `--input <dir>` (Required): Source directory containing documents (PDF, Images, DOCX).
*   `--output <dir>` (Required): Output directory for processed Markdown files.
*   `--vision-model <name>`: Vision model to use in LM Studio (default: `qwen2-vl-7b-instruct`).
*   `--pdf-provider <name>`: Provider for PDF OCR (default: `lm-studio`).
*   `--pdf-model <name>`: Model for PDF OCR (default: `qwen/qwen3-vl-30b`).

### `rag-chat` (Chat & Index)
Starts the RAG interface.
*   `--input <dir>` (Required): Directory containing processed Markdown files.
*   `--db <dir>` (Required): Path to ChromaDB directory.
*   `--index-only`: Run indexing only and exit.
*   `--model <name>`: Ollama LLM model name (default: `gemma3:12b`).
*   `--embedding-model <name>`: Ollama embedding model name (default: `embeddinggemma:latest`).

### `rag-add` (Add Single Document)
Adds, processes, and indexes a single file.
*   `file_path`: Path to the file to add.
*   `--source-dir <dir>` (Required): Directory where the file will be copied.
*   `--processed-dir <dir>`: Output directory for Markdown (default: `./processed_md`).
*   `--db <dir>`: Path to ChromaDB directory (default: `./chroma_db`).

---

## 💾 Permanent Installation (Optional)

If you use this tool frequently, install it globally:

```bash
uv tool install git+https://github.com/laurentvv/rag-ollama
```

Now you can run commands directly:

**Prepare:**
```bash
rag-prepare --input "C:\MyDocs" --output "./processed_md"
```

**Chat:**
```bash
rag-chat --input "./processed_md" --db "./chroma_db"
```

**Add Document:**
```bash
rag-add "C:\MyDocs\new.pdf" --source-dir "C:\MyDocs"
```

## 🤖 Model Configuration

### Ollama Models
By default, the system uses `gemma3:12b` and `embeddinggemma:latest`. You can change this via arguments:
```bash
rag-chat --input "./processed_md" --db "./chroma_db" --model "llama3" --embedding-model "nomic-embed-text"
```

### LM Studio (Vision)
By default, the system uses `qwen2-vl-7b-instruct` for image analysis. You can change this during preparation:
```bash
rag-prepare --input "C:\MyDocs" --output "./processed_md" --vision-model "llama-3.2-vision"
```

## 🤖 Setup AI Providers

### Setup Ollama
1.  Download and install [Ollama](https://ollama.com/).
2.  Pull the required models:
    ```bash
    ollama pull gemma3:12b
    ollama pull embeddinggemma:latest
    ```

### Setup LM Studio (for Vision/OCR)
1.  Download and install [LM Studio](https://lmstudio.ai/).
2.  Load a Vision model (e.g., `qwen2-vl` or `llama-3.2-vision`).
3.  Go to the **Developer/Server** tab (double arrow icon).
4.  Start the local server on port **1234**.
    *   Ensure "Cross-Origin-Resource-Sharing (CORS)" is enabled (usually on by default).

## 🛠️ How It Works

```mermaid
graph TD
    subgraph "Document Ingestion"
        A[Source Docs] --> B{File Type?}
        B -- PDF --> C[pdf-ocr-ai (VLM)]
        B -- Image --> C
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

## 📂 Project Structure

*   `src/rag_ollama/`: Source code.
*   `tests/`: Unit and integration tests.
*   `processed_md/`: Directory containing the generated Markdown files.
*   `chroma_db/`: Local vector database storage.
*   `pyproject.toml`: Project configuration and dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.