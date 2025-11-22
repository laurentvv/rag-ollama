from pathlib import Path
import subprocess
import os
from .base import DocumentProcessor
from ..config import RAGConfig
from ..utils.exceptions import ProcessingError

class PDFProcessor(DocumentProcessor):
    """Processor for PDF files using pdf-ocr-ai."""

    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def process(self, file_path: Path, config: RAGConfig) -> Path:
        output_path = config.processed_dir / (file_path.stem + ".md")
        command = [
            "pdf-ocr-ai", str(file_path), str(output_path),
            "--provider", "ollama", "--model", config.pdf_model
        ]
        try:
            env = os.environ.copy()
            env["OPENAI_TIMEOUT"] = "600"
            subprocess.run(command, check=True, encoding='utf-8', errors='replace', env=env, capture_output=True)
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ProcessingError(f"Erreur lors du traitement PDF de {file_path.name}: {e}")
