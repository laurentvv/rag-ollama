from pathlib import Path
import ollama
from .base import DocumentProcessor
from ..config import RAGConfig
from ..utils.exceptions import ProcessingError

class ImageProcessor(DocumentProcessor):
    """Processor for image files using Ollama's vision models."""

    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]

    def process(self, file_path: Path, config: RAGConfig) -> Path:
        output_path = config.processed_dir / (file_path.stem + ".md")
        try:
            client = ollama.Client(host='http://localhost:11434', timeout=600)
            response = client.chat(
                model=config.vision_model,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail, capturing all visible text and visual elements for documentation purposes.',
                    'images': [str(file_path)]
                }],
                options={'num_ctx': 4096},
                keep_alive='15m'
            )
            output_path.write_text(response['message']['content'], encoding='utf-8')
            return output_path
        except Exception as e:
            raise ProcessingError(f"Erreur lors du traitement d'image de {file_path.name}: {e}")
