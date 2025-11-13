# Qwen Code - RAG-Ollama Project Context

## Project Overview

This is a **Simple Ollama CLI** project - a minimal implementation for indexing, searching, and asking questions with Ollama integration. It's a Retrieval-Augmented Generation (RAG) system that uses Ollama models for both embeddings and language model responses.

The project provides a command-line interface with three main operations:
- `build`: Create an embedding index from text or files
- `search`: Perform semantic search in the index
- `ask`: Ask questions about indexed content using a language model

## Architecture & Components

The project consists of:
1. **SimpleEmbeddingIndex**: Handles indexing of text, computation of embeddings, and similarity search
2. **SimpleCLI**: Provides the command-line interface for build, search, and ask commands
3. **Ollama Integration**: Uses Ollama for both embedding computation and LLM responses
4. **File-based Storage**: Stores indexed data in JSONL, pickle, and JSON files

## Dependencies

Key Python dependencies (from requirements.txt):
- `numpy>=1.21.0`: For numerical operations
- `ollama>=0.1.0`: For Ollama API integration
- `tqdm>=4.64.0`: For progress bars during embedding computation
- `PyPDF2>=3.0.0`: For PDF document parsing
- `markdown>=3.0.0`: For Markdown document parsing
- `python-docx>=0.8.0`: For Microsoft Word document parsing

## Required Ollama Models

The system requires Ollama to be running locally with these models:
- **Embedding model**: `nomic-embed-text` (default) - for text embeddings
- **LLM model**: `llama3.2:1b` (default) or other Ollama LLMs - for question answering

## File Structure

After building an index, the system creates three files:
- `index_name_passages.jsonl` - Contains the indexed passages
- `index_name_embeddings.pkl` - Contains the computed embeddings  
- `index_name_meta.json` - Contains index metadata

## Building and Running

### Prerequisites
1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage Examples

1. **Build an Index**:
   ```bash
   python simple_ollama_cli.py build my_index.leann --text "Paris is the capital of France. Tokyo is the capital of Japan."
   ```

2. **Build from a File**:
   ```bash
   python simple_ollama_cli.py build my_index.leann --file document.txt
   ```

3. **Build from a Directory** (NEW!):
   ```bash
   python simple_ollama_cli.py build my_index.leann --directory /path/to/documents/
   ```

The tool supports multiple document formats:
- Text files (.txt)
- Markdown files (.md)
- PDF files (.pdf)
- Microsoft Word documents (.docx)

4. **Search in an Index**:
   ```bash
   python simple_ollama_cli.py search my_index.leann "European capitals"
   ```

5. **Ask Questions**:
   ```bash
   python simple_ollama_cli.py ask my_index.leann "What are European capitals?" --llm-model llama3.2:1b
   ```

## Key Configuration Options

### Build Command Options:
- `--text`: Direct text to index
- `--file`: Single file to index
- `--directory`: Directory to index (recursively searches for .txt, .md, .pdf, .docx files)
- `--embedding-model`: Ollama embedding model (default: nomic-embed-text)
- `--chunk-size`: Chunk size for text splitting (default: 512)

### Search Command Options:
- `--top-k`: Number of results to return (default: 5)
- `--embedding-model`: Ollama embedding model (default: nomic-embed-text)

### Ask Command Options:
- `--top-k`: Number of passages to retrieve (default: 5)
- `--llm-model`: Ollama LLM model (default: llama3.2:1b)
- `--embedding-model`: Ollama embedding model (default: nomic-embed-text)

## Text Processing

The system includes:
- **Text Chunking**: Automatically chunks long documents using a simple word-based approach
- **Normalization**: Uses cosine similarity for search ranking
- **Progress Tracking**: Shows progress with tqdm during embedding computation

## Current State

The project includes:
- Example document (example_document.txt) with sample content about capitals
- Test index files demonstrating the output format
- Complete implementation with error handling
- Command-line interface with subcommands
- Support for both direct text input and file-based indexing

## Development Conventions

- Use type hints throughout the codebase
- Include docstrings for classes and methods
- Use pathlib for path handling
- Follow Python 3.7+ standards
- Include progress indicators for long-running operations
- Use JSON for metadata and JSONL for passages storage

## Testing & Validation

The project includes test index files (test_index_*.*) that demonstrate the expected output format and can be used for validation. The example_document.txt serves as a sample input for testing the functionality.

## Qwen Added Memories
- PROD1
