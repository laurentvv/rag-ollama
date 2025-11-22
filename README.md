# 🧠 RAG-Ollama

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Integrated-orange)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](https://github.com/laurentvv/rag-ollama/blob/main/LICENSE)

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
*   **[Ollama](https://ollama.com/)**: Running locally for LLM, Embeddings, and Vision.

## 🚀 Workflow (Quick Start)

You don't need to install anything! Just ensure you have the **Prerequisites** ready, then follow these steps using `uvx`.

### 1️⃣ Step 1: Prepare Documents
**Goal:** Convert your raw documents (PDF, Images, DOCX) into clean, machine-readable Markdown.

*   **Why?** LLMs work best with plain text. We use AI-powered OCR to "read" your PDFs and images and convert them into a structured text format that the system can easily understand and index.

```bash
# Process all documents in the source directory
uvx --from git+https://github.com/laurentvv/rag-ollama rag-prepare --input "C:\MyDocs" --output "./processed_md"
```

### 2️⃣ Step 2: Chat & Index
**Goal:** Index the processed text and start the conversation.

*   **Why?** The system needs to "learn" your documents by converting them into mathematical vectors (indexing) and storing them in a local database (ChromaDB). This allows it to instantly find the most relevant information when you ask a question.

```bash
uvx --from git+https://github.com/laurentvv/rag-ollama rag-chat --input "./processed_md" --db "./chroma_db"
```

### ⚡ Quick Add (Single File)
To add a single document without reprocessing everything:

```bash
uvx --from git+https://github.com/laurentvv/rag-ollama rag-add "C:\MyDocs\new_file.pdf" --source-dir "C:\MyDocs" --processed-dir "./processed_md" --db "./chroma_db"

### `rag-prepare` Arguments
| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input <dir>` | ✅ | - | Source directory containing documents. |
| `--output <dir>` | ✅ | - | Output directory for processed Markdown. |
| `--vision-model <name>` | ❌ | `llama3.2-vision` | Vision model to use in Ollama for images. |
| `--pdf-model <name>` | ❌ | `llama3.2-vision` | Model for PDF OCR. |

### `rag-chat` (Chat & Index)
Starts the RAG interface.

| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input <dir>` | ✅ | - | Directory containing processed Markdown files. |
| `--db <dir>` | ✅ | - | Path to ChromaDB directory. |
| `--index-only` | ❌ | `False` | Run indexing only and exit. |
| `--model <name>` | ❌ | `gemma3:12b` | Ollama LLM model name. |
| `--embedding-model <name>` | ❌ | `embeddinggemma:latest` | Ollama embedding model name. |

### `rag-add` (Add Single Document)
Adds, processes, and indexes a single file.

| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `file_path` | ✅ | - | Path to the file to add. |
| `--source-dir <dir>` | ✅ | - | Directory where the file will be copied. |
| `--processed-dir <dir>` | ❌ | `./processed_md` | Output directory for Markdown. |
| `--db <dir>` | ❌ | `./chroma_db` | Path to ChromaDB directory. |

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

## 🤖 Model Configuration & Selection

### Understanding the Models
This system uses three types of AI models, all running locally via Ollama:

1.  **LLM (Large Language Model)**: The "brain" that answers your questions (e.g., `gemma3:12b`, `llama3`, `mistral`).
2.  **Embedding Model**: Converts text into numbers (vectors) for search (e.g., `embeddinggemma:latest`, `nomic-embed-text`).
3.  **Vision Model**: Analyzes images and PDFs for OCR (e.g., `llama3.2-vision`, `qwen2.5-vl`).

### How to Find and Choose Models
You can browse available models at **[ollama.com/library](https://ollama.com/library)**.

*   **For General Chat (LLM)**: Look for models with high reasoning capabilities.
    *   *Recommended*: `gemma3:12b` (balanced), `llama3:8b` (fast), `mistral` (good generalist).
*   **For Embeddings**: Look for models specifically trained for embeddings.
    *   *Recommended*: `embeddinggemma` (Google's latest), `nomic-embed-text` (very popular for RAG).
*   **For Vision/OCR**: Look for "multimodal" or "vision" models.
    *   **[Browse Ollama Vision Models](https://ollama.com/search?c=vision)**
    *   *Recommended*: `llama3.2-vision` (excellent), `qwen2.5-vl` (strong performance), `llava` (classic).
    *   *Note*: `pdf-ocr-ai` uses these models to "see" the document content. Larger models (like `qwen2.5-vl:32b`) are more accurate but slower.

### Changing Models via CLI
You can override the default models using command-line arguments:

**Chat with a different LLM and Embedding model:**
```bash
rag-chat --input "./processed_md" --db "./chroma_db" --model "mistral" --embedding-model "nomic-embed-text"
```

**Prepare documents with a different Vision model:**
```bash
rag-prepare --input "C:\MyDocs" --output "./processed_md" --vision-model "llava" --pdf-model "llava"
```

## 🤖 Setup AI Providers

### Setup Ollama
1.  Download and install [Ollama](https://ollama.com/).
2.  **Pull the default models** (recommended for first run):
    ```bash
    ollama pull gemma3:12b
    ollama pull embeddinggemma:latest
    ollama pull llama3.2-vision
    ```
3.  **Pull alternative models** (if you want to customize):
    ```bash
    ollama pull mistral
    ollama pull nomic-embed-text
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
