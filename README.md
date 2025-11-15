# Simple Ollama CLI

A minimal implementation for indexing, searching, and asking with Ollama integration, based on LEANN concepts but greatly simplified.

## Overview

This tool provides three main commands:
- `build`: Create an embedding index from text or files
- `search`: Perform semantic search in the index
- `ask`: Ask questions about indexed content using a language model

## Requirements

- Python 3.7+
- Ollama running locally
- Required models: `nomic-embed-text` (for embeddings) and a LLM like `llama3.2`

## Prerequisites

### Tesseract-OCR (for Document Preprocessing)

The `prepare_documents.py` script uses `docling` with an OCR (Optical Character Recognition) feature to extract text from images and screenshots within your documents. This requires the Tesseract-OCR engine to be installed on your system.

**Please install Tesseract-OCR before running `prepare_documents.py`.**

#### Installation Instructions:

*   **Windows**:
    *   Download the official installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) repository.
    *   During installation, ensure you select the option to **"Add Tesseract to the system PATH"**.

*   **Linux (Debian/Ubuntu)**:
    ```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    # Install language packs if needed (e.g., for French)
    sudo apt-get install tesseract-ocr-fra
    ```

*   **macOS (via Homebrew)**:
    ```bash
    brew install tesseract
    ```

To verify the installation, open a new terminal and run `tesseract --version`. If it shows the version number, you are all set.

## Installation

1. **Install Ollama** from [ollama.com](https://ollama.com)

2. **Pull required models**:
   ```bash
   ollama pull nomic-embed-text  # for embeddings (required)
   ollama pull llama3.2:1b       # for question answering (lightweight, recommended)
   # OR
   ollama pull llama3.2          # for question answering (larger, more capable)
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Build an Index

Create an index from direct text:
```bash
python simple_ollama_cli.py build my_index.leann --text "Paris is the capital of France. Tokyo is the capital of Japan."
```

Or build from a file:
```bash
python simple_ollama_cli.py build my_index.leann --file document.txt
```

The tool now supports multiple document formats including:
- Text files (.txt)
- Markdown files (.md)
- PDF files (.pdf)
- Microsoft Word documents (.docx)

Or build from a directory (recursively indexes all .txt, .md, .pdf, .docx files):
```bash
python simple_ollama_cli.py build my_index.leann --directory /path/to/documents/
```

You can specify custom models:
```bash
python simple_ollama_cli.py build my_index.leann --file document.txt --embedding-model nomic-embed-text
```

### Search in an Index

Search for content in your index:
```bash
python simple_ollama_cli.py search my_index.leann "European capitals"
```

Customize search parameters:
```bash
python simple_ollama_cli.py search my_index.leann "European capitals" --top-k 3 --embedding-model nomic-embed-text
```

### Ask Questions

Ask questions about indexed content:
```bash
python simple_ollama_cli.py ask my_index.leann "What are European capitals?" --llm-model llama3.2:1b
```

Customize both models:
```bash
python simple_ollama_cli.py ask my_index.leann "What are European capitals?" \
  --llm-model llama3.2:1b \
  --embedding-model nomic-embed-text
```

## Command Reference

### `build` command options:
- `index_path`: Path to save the index (required)
- `--text`: Direct text to index
- `--file`: File to index
- `--directory`: Directory to index (recursively searches for .txt, .md, .pdf, .docx files)
- `--embedding-model`: Ollama embedding model to use (default: nomic-embed-text)
- `--chunk-size`: Chunk size for text splitting (default: 512)

### `search` command options:
- `index_path`: Path to the index (required)
- `query`: Search query (required)
- `--top-k`: Number of results to return (default: 5)
- `--embedding-model`: Ollama embedding model to use (default: nomic-embed-text)

### `ask` command options:
- `index_path`: Path to the index (required)
- `query`: Question to ask (required)
- `--top-k`: Number of passages to retrieve (default: 5)
- `--llm-model`: Ollama LLM model to use (default: llama3.2:1b)
- `--embedding-model`: Ollama embedding model to use (default: nomic-embed-text)

## Models to Use

### Embedding Models (for `--embedding-model`):
- `nomic-embed-text` - Fast and efficient (default)
- `mxbai-embed-large` - High quality
- `all-minilm` - Good balance of speed and quality

### LLM Models (for `--llm-model`):
- `llama3.2:1b` - Lightweight, fast (recommended for resource-constrained systems)
- `llama3.2:3b` - Medium size, balanced performance
- `llama3.2` - Full size, most capable
- `mistral` - Alternative option
- `gemma2` - Google's efficient model

## File Structure

After building an index, you'll have these files:
- `index_name_passages.jsonl` - Contains the indexed passages
- `index_name_embeddings.pkl` - Contains the computed embeddings
- `index_name_meta.json` - Contains index metadata

## Example Workflow

1. Create an index:
   ```bash
   python simple_ollama_cli.py build travel_info.leann --text "Paris is the capital of France and is known for its art and culture. Tokyo is the capital of Japan and is famous for its technology and traditions. Rome is the capital of Italy and is known for its ancient history and architecture."
   ```

2. Search for information:
   ```bash
   python simple_ollama_cli.py search travel_info.leann "European capitals"
   ```

3. Ask questions:
   ```bash
   python simple_ollama_cli.py ask travel_info.leann "What cities are known for their art and culture?" --llm-model llama3.2:1b
   ```

## Implementation Details

This simplified implementation contains the core functionality of LEANN with Ollama integration:

- **SimpleEmbeddingIndex**: Handles indexing of text, computation of embeddings, and similarity search
- **SimpleCLI**: Provides the command-line interface for build, search, and ask commands
- **Ollama Integration**: Uses Ollama for both embedding computation and LLM responses
- **File-based Storage**: Stores indexed data in JSONL, pickle, and JSON files

The implementation is designed to be:
- Easy to understand and modify
- Lightweight with minimal dependencies
- Compatible with Ollama's API

## How Search and Ask Work

### Search Functionality

The `search` command performs semantic search based on vector similarity:

1. **Query Embedding**: Your search query is converted to an embedding vector using the same model that was used to create the index (default: `nomic-embed-text`)
2. **Similarity Calculation**: Cosine similarity is computed between your query embedding and all embeddings in the index
3. **Top-k Retrieval**: The system retrieves the top-k most similar passages (default: 5, customizable with `--top-k`)
4. **Results**: Results are returned with similarity scores indicating how closely each passage matches your query

The search uses the same embedding model for both indexing and querying to ensure consistency in the vector space.

### Ask Functionality

The `ask` command implements Retrieval-Augmented Generation (RAG) to answer questions:

1. **Retrieval Phase**:
   - Uses the `search` functionality to find the most relevant passages (default: 5, customizable with `--top-k`)
   - These passages serve as context for the LLM

2. **Generation Phase**:
   - Builds a prompt containing the retrieved passages as context
   - Example prompt structure:
     ```
     Based on the following context, please answer the question:

     Context:
     [Most relevant passages from your documents]

     Question: [Your question]

     Answer:
     ```
   - Sends this prompt to the specified LLM (default: `llama3.2:1b`, customizable with `--llm-model`)

3. **Response**: The LLM generates an answer based on the provided context, which is then returned to you

### Understanding Top-k Parameter

The `--top-k` parameter controls how many passages are retrieved and used as context:

- **For search**: Controls how many results are returned
- **For ask**: Controls how many passages are included in the context prompt

Recommended values:
- Small models (8k context): `--top-k 3-7`
- Medium models (32k context, like Qwen3): `--top-k 5-15`
- Large models (128k+ context): `--top-k 10-20`

Start with the default (5) and adjust based on your needs. More passages provide more context but consume more tokens and may introduce noise.

### Model Usage During Operations

- **Build**: Uses embedding model to convert text to vectors
- **Search**: Uses the same embedding model to convert your query to a vector and find similar passages
- **Ask**: Uses both embedding model (for retrieval) and LLM (for generation)

## Dependencies

- `numpy`: For numerical operations
- `ollama`: For Ollama API integration
- `tqdm`: For progress bars during embedding computation

## Notes

- The embedding model is used for both indexing (building) and searching
- Choose embedding models based on your needs for speed vs quality
- The LLM model is only used for the `ask` command
- For large documents, the tool automatically chunks text using the `--chunk-size` parameter
- The search functionality works well even if the LLM functionality has issues
- Use the smaller `llama3.2:1b` model for better performance on resource-constrained systems