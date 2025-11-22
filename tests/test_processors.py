import pytest
from pathlib import Path
from unittest.mock import patch
from src.rag_ollama.config import RAGConfig
from src.rag_ollama.processors.pdf import PDFProcessor
from src.rag_ollama.processors.image import ImageProcessor
from src.rag_ollama.processors.docling import DoclingProcessor

@pytest.fixture
def config(tmp_path):
    """Provides a RAGConfig instance with temporary paths."""
    source_dir = tmp_path / "source"
    processed_dir = tmp_path / "processed"
    db_dir = tmp_path / "db"
    source_dir.mkdir()
    processed_dir.mkdir()
    db_dir.mkdir()
    return RAGConfig(
        source_dir=source_dir,
        processed_dir=processed_dir,
        db_path=db_dir,
    )

def test_pdf_processor(config, tmp_path):
    """Tests the PDFProcessor's ability to handle .pdf files."""
    processor = PDFProcessor()
    pdf_file = config.source_dir / "test.pdf"
    pdf_file.touch()

    assert processor.can_process(pdf_file)
    assert not processor.can_process(config.source_dir / "test.docx")

    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        processor.process(pdf_file, config)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert str(pdf_file) in call_args
        assert str(config.processed_dir / "test.md") in call_args

def test_image_processor(config, tmp_path, mock_ollama):
    """Tests the ImageProcessor's ability to handle image files."""
    processor = ImageProcessor()
    img_file = config.source_dir / "test.jpg"
    img_file.touch()

    assert processor.can_process(img_file)
    assert not processor.can_process(config.source_dir / "test.txt")

    output_path = processor.process(img_file, config)
    assert output_path.exists()
    assert "Mocked AI response" in output_path.read_text()
    mock_ollama.return_value.chat.assert_called_once()

def test_docling_processor(config, tmp_path):
    """Tests the DoclingProcessor's ability to handle DOCX files."""
    processor = DoclingProcessor()
    docx_file = config.source_dir / "test.docx"
    docx_file.touch()

    assert processor.can_process(docx_file)
    assert not processor.can_process(config.source_dir / "test.pdf")

    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        processor.process(docx_file, config)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert str(docx_file) in call_args
        assert "--no-ocr" in call_args # No images in a blank docx
