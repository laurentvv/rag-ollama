# RAG-Ollama Project Context

## Project Overview

RAG-Ollama is an advanced Local Retrieval-Augmented Generation (RAG) system built with Python 3.10+ that leverages Ollama for local AI processing. The system combines vector search capabilities with keyword-based retrieval to provide intelligent document processing and question-answering capabilities. It features AI-powered OCR for PDFs, image processing with vision models, and hybrid search combining semantic vectors with BM25 keyword search.

### Key Features
- **Fully Local Processing**: Runs entirely on your machine using Ollama and local embeddings
- **Smart AI OCR**: Uses Vision Language Models (VLMs) via `pdf-ocr-ai` to process PDFs and images
- **Multi-Format Support**: Handles PDF, DOCX, PPTX, images, and Markdown files
- **Hybrid Search**: Combines ChromaDB vector search with BM25 keyword search for superior retrieval
- **Incremental Indexing**: Tracks document changes using file hashing to update only modified content

### Architecture
The system consists of three main components:
1. **Document Preparation** (`prepare_documents.py`): Converts source documents to processed Markdown
2. **Chat & Indexing** (`rag.py`): Handles vector database operations and question-answering
3. **Document Addition** (`add_document.py`): Adds single documents to an existing index

### Core Dependencies
- `ollama`: Client for interacting with local Ollama instances
- `langchain`: Framework for building language model applications
- `chromadb`: Vector database for document storage and retrieval
- `pdf-ocr-ai`: Custom dependency for AI-powered PDF OCR (git source)
- `rank_bm25`: BM25 retrieval algorithm for keyword-based search
- `sentence-transformers`: For text embeddings

## Building and Running

### Prerequisites
- Python 3.10+ with `uv` package manager
- Ollama running locally
- Default models: `gemma3:12b`, `embeddinggemma:latest`, `llama3.2-vision`

### Installation
```bash
# Install using uv for best experience
uv tool install git+https://github.com/laurentvv/rag-ollama

# Or run directly without installation
uvx --from git+https://github.com/laurentvv/rag-ollama
```

### Usage Workflow
1. **Prepare Documents**: Convert source documents to processed Markdown
   ```bash
   rag-prepare --input "./docs" --output "./processed_md"
   ```

2. **Chat & Index**: Index processed documents and start Q&A session
   ```bash
   rag-chat --input "./processed_md" --db "./chroma_db"
   ```

3. **Add Single Document**: Add one document to existing index
   ```bash
   rag-add "path/to/document.pdf" --source-dir "./docs"
   ```

### Default Models Configuration
- LLM: `gemma3:12b`
- Embedding: `embeddinggemma:latest`
- Vision: `llama3.2-vision`

## Development Conventions

### Configuration
- Configuration is managed through `config.yaml` and command-line arguments
- Default values can be overridden via CLI parameters
- The `RAGConfig` dataclass centralizes all configuration parameters

### File Processing Pipeline
1. Processors in `src/rag_ollama/processors/` handle different file types
2. Each processor implements `can_process()` and `process()` methods
3. Processed files are stored as Markdown with source context headers
4. File hashing ensures incremental updates only process changed content

### Testing
- Unit tests in `tests/` directory using pytest
- Mock-based testing for database and API interactions
- Focus on incremental indexing logic and processor functionality

### Error Handling
- Custom exception classes in `utils/exceptions.py`
- Logging through `utils/logging.py` with structured messages
- Graceful handling of missing models, Ollama unavailability, and file processing errors

## Project Structure
```
src/rag_ollama/
├── config.py          # Configuration dataclass
├── rag.py            # Main RAG system (chat & indexing)
├── prepare_documents.py # Document preprocessing
├── add_document.py   # Add single documents
├── processors/       # File type specific processors
│   ├── base.py       # Base processor interface
│   ├── pdf.py        # PDF processing with pdf-ocr-ai
│   ├── image.py      # Image processing with VLMs
│   └── docling.py    # Additional document processing
└── utils/            # Utilities and helpers
```

## Special Considerations
- The system uses `pdf-ocr-ai` as a git dependency for advanced PDF processing
- Image and PDF processing rely on Vision Language Models running locally
- Hash-based incremental indexing prevents unnecessary reprocessing
- Hybrid search (ChromaDB + BM25) provides robust document retrieval
- The system is designed to work entirely offline once models are pulled