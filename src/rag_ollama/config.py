# Nouveau fichier: src/rag_ollama/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RAGConfig:
    source_dir: Path
    processed_dir: Path
    db_path: Path
    vision_model: str = "llama3.2-vision"
    pdf_model: str = "llama3.2-vision"
    llm_model: str = "gemma3:12b"
    embedding_model: str = "embeddinggemma:latest"
