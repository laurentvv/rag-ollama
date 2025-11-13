#!/usr/bin/env python3
"""
Simple CLI for indexing, searching and asking with Ollama integration.
Minimal implementation based on LEANN concepts.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
import ollama
from tqdm import tqdm

# Optional imports for document parsing - these will be checked at runtime
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import markdown
    MD_SUPPORT = True
except ImportError:
    MD_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

class SimpleEmbeddingIndex:
    """Simple embedding index using Ollama for embeddings."""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_dir = self.index_path.parent
        self.passages_file = self.index_dir / f"{self.index_path.stem}_passages.jsonl"
        self.embeddings_file = self.index_dir / f"{self.index_path.stem}_embeddings.pkl"
        self.meta_file = self.index_dir / f"{self.index_path.stem}_meta.json"
        
        # Create directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.passages: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a text passage to be indexed."""
        if metadata is None:
            metadata = {}
        passage_id = str(len(self.passages))
        passage = {
            "id": passage_id,
            "text": text,
            "metadata": metadata
        }
        self.passages.append(passage)
    
    def compute_embeddings(self, texts: List[str], model: str = "nomic-embed-text") -> np.ndarray:
        """Compute embeddings using Ollama."""
        print(f"Computing embeddings for {len(texts)} passages...")
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            response = ollama.embeddings(model=model, prompt=text)
            embedding = np.array(response['embedding'])
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def build_index(self, embedding_model: str = "nomic-embed-text"):
        """Build the embedding index."""
        if not self.passages:
            raise ValueError("No passages added to index.")
        
        # Compute embeddings
        texts = [p["text"] for p in self.passages]
        self.embeddings = self.compute_embeddings(texts, embedding_model)
        
        # Save passages to JSONL file
        with open(self.passages_file, 'w', encoding='utf-8') as f:
            for passage in self.passages:
                f.write(json.dumps(passage) + '\n')
        
        # Save embeddings to pickle file
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save metadata
        meta_data = {
            "embedding_model": embedding_model,
            "num_passages": len(self.passages),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0
        }
        
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"Index built successfully at {self.index_path}")
    
    def load_index(self):
        """Load the embedding index from disk."""
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Index metadata not found: {self.meta_file}")
        
        # Load metadata
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # Load passages
        self.passages = []
        with open(self.passages_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.passages.append(json.loads(line))
        
        # Load embeddings
        with open(self.embeddings_file, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        print(f"Index loaded successfully from {self.index_path}")
    
    def search(self, query: str, top_k: int = 5, embedding_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
        """Search for similar passages to the query."""
        if self.embeddings is None:
            raise ValueError("Index not loaded or built.")
        
        # Compute query embedding
        query_response = ollama.embeddings(model=embedding_model, prompt=query)
        query_embedding = np.array(query_response['embedding']).reshape(1, -1)
        
        # Calculate cosine similarity
        # Normalize embeddings
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(normalized_embeddings, normalized_query.T).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            passage = self.passages[idx]
            results.append({
                "id": passage["id"],
                "text": passage["text"],
                "metadata": passage["metadata"],
                "score": score
            })
        
        return results


class SimpleCLI:
    """Simple CLI for build, search, and ask commands."""
    
    def __init__(self):
        pass
    
    def build(self, args):
        """Build an index from text passages."""
        index = SimpleEmbeddingIndex(args.index_path)

        if args.text:
            # Add text from command line
            index.add_text(args.text)
        elif args.file:
            # Add text from file
            file_path = Path(args.file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple chunking for long documents
                chunks = self._chunk_text(content, args.chunk_size)
                for chunk in chunks:
                    index.add_text(chunk)
        elif args.directory:
            # Add text from all supported document types in directory and subdirectories
            dir_path = Path(args.directory)
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            # Find all supported files in the directory and subdirectories
            supported_files = []
            for ext in ['*.txt', '*.md', '*.pdf', '*.docx']:
                supported_files.extend(list(dir_path.rglob(ext)))

            if not supported_files:
                print(f"No supported files found in {dir_path} (supported: .txt, .md, .pdf, .docx)")
                return

            print(f"Found {len(supported_files)} supported files to index")

            for file_path in supported_files:
                print(f"Processing {file_path}...")
                try:
                    content = self._read_file_content(file_path)
                    if content:
                        # Simple chunking for long documents
                        chunks = self._chunk_text(content, args.chunk_size)
                        for chunk in chunks:
                            # Add metadata with the file path and type
                            metadata = {
                                "source_file": str(file_path.relative_to(dir_path)),
                                "file_type": file_path.suffix.lower()
                            }
                            index.add_text(chunk, metadata=metadata)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
        else:
            print("No input provided. Please specify text, file, or directory to index.")
            return

        index.build_index(embedding_model=args.embedding_model)
    
    def search(self, args):
        """Search in an existing index."""
        index = SimpleEmbeddingIndex(args.index_path)
        index.load_index()
        
        results = index.search(
            query=args.query,
            top_k=args.top_k,
            embedding_model=args.embedding_model
        )
        
        print(f"\nSearch results for '{args.query}':")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:200]}...")
            if result['metadata']:
                print(f"   Metadata: {result['metadata']}")
            print()
    
    def ask(self, args):
        """Ask questions using the indexed content and Ollama LLM."""
        index = SimpleEmbeddingIndex(args.index_path)
        index.load_index()
        
        results = index.search(
            query=args.query,
            top_k=args.top_k,
            embedding_model=args.embedding_model
        )
        
        if not results:
            print("No relevant content found.")
            return
        
        # Build context from search results
        context_parts = []
        for result in results:
            context_parts.append(result['text'])
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Ollama
        prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {args.query}

Answer:"""
        
        print(f"\nUsing context from {len(results)} passages...")
        print(f"Context preview: {context[:300]}...")
        
        # Generate response using Ollama
        response = ollama.chat(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(f"\nAnswer from {args.llm_model}:")
        print(response['message']['content'])
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read content from different file types."""
        file_ext = file_path.suffix.lower()

        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == '.md':
            if not MD_SUPPORT:
                print(f"Warning: markdown module not available, skipping {file_path}")
                return ""
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Convert markdown to plain text by stripping formatting
                # Just return the raw content for now, but in the future we might want to strip markdown
                return content
        elif file_ext == '.pdf':
            if not PDF_SUPPORT:
                print(f"Warning: PyPDF2 module not available, skipping {file_path}")
                return ""
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        elif file_ext == '.docx':
            if not DOCX_SUPPORT:
                print(f"Warning: python-docx module not available, skipping {file_path}")
                return ""
            doc = Document(file_path)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return content
        else:
            # For unsupported file types, try to read as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                print(f"Could not read file {file_path} as text")
                return ""

    def _chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks


def main():
    parser = argparse.ArgumentParser(
        prog="simple-ollama-cli",
        description="Simple CLI for indexing, searching and asking with Ollama"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build an index")
    build_parser.add_argument("index_path", help="Path to save the index")
    build_parser.add_argument("--text", help="Direct text to index")
    build_parser.add_argument("--file", help="File to index")
    build_parser.add_argument("--directory", help="Directory to index (recursively searches for .txt files)")
    build_parser.add_argument("--embedding-model", default="nomic-embed-text",
                             help="Ollama embedding model to use (default: nomic-embed-text)")
    build_parser.add_argument("--chunk-size", type=int, default=512,
                             help="Chunk size for text splitting (default: 512)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search in an index")
    search_parser.add_argument("index_path", help="Path to the index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5,
                              help="Number of results to return (default: 5)")
    search_parser.add_argument("--embedding-model", default="nomic-embed-text",
                              help="Ollama embedding model to use (default: nomic-embed-text)")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask questions about indexed content")
    ask_parser.add_argument("index_path", help="Path to the index")
    ask_parser.add_argument("query", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5,
                           help="Number of passages to retrieve (default: 5)")
    ask_parser.add_argument("--llm-model", default="llama3.2",
                           help="Ollama LLM model to use (default: llama3.2)")
    ask_parser.add_argument("--embedding-model", default="nomic-embed-text",
                           help="Ollama embedding model to use (default: nomic-embed-text)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = SimpleCLI()
    
    if args.command == "build":
        cli.build(args)
    elif args.command == "search":
        cli.search(args)
    elif args.command == "ask":
        cli.ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()