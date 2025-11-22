from pathlib import Path
import subprocess
import sys
import zipfile
from .base import DocumentProcessor
from ..config import RAGConfig
from ..utils.exceptions import ProcessingError

class DoclingProcessor(DocumentProcessor):
    """Processor for various document types using Docling."""

    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".docx", ".pptx", ".html", ".asciidoc", ".txt", ".md"]

    def _has_images_docx(self, file_path: Path) -> bool:
        try:
            if file_path.suffix.lower() in [".docx", ".pptx"]:
                with zipfile.ZipFile(file_path, 'r') as z:
                    return any("media/" in name and not name.endswith("/") for name in z.namelist())
            return False
        except Exception:
            return False

    def process(self, file_path: Path, config: RAGConfig) -> Path:
        output_path = config.processed_dir / (file_path.stem + ".md")
        use_ocr = self._has_images_docx(file_path)

        python_dir = Path(sys.executable).parent
        docling_cmd = str(python_dir / "docling.exe") if (python_dir / "docling.exe").exists() else "docling"

        command = [
            docling_cmd, str(file_path), "--to", "md",
            "--output", str(config.processed_dir)
        ]
        if use_ocr:
            command.extend(["--ocr", "--enrich-picture-description"])
        else:
            command.extend(["--no-ocr", "--no-enrich-picture-description"])

        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            return output_path
        except subprocess.CalledProcessError as e:
            raise ProcessingError(f"Erreur Docling sur {file_path.name}: {e.stderr}")
