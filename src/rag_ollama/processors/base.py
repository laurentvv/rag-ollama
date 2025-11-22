from abc import ABC, abstractmethod
from pathlib import Path
from ..config import RAGConfig

class DocumentProcessor(ABC):
    """Abstract base class for a document processor."""

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Whether this processor can handle the given file type."""
        pass

    @abstractmethod
    def process(self, file_path: Path, config: RAGConfig) -> Path:
        """Processes the file and returns the path to the output Markdown."""
        pass
