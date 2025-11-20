import pytest
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_ollama.prepare_documents import has_images_docx, process_image_with_ai

def test_has_images_docx_no_images(tmp_path):
    # Create a dummy zip file (docx structure) without images
    d = tmp_path / "test.docx"
    import zipfile
    with zipfile.ZipFile(d, 'w') as z:
        z.writestr('word/document.xml', 'content')
    
    assert has_images_docx(d) == False

def test_has_images_docx_with_images(tmp_path):
    # Create a dummy zip file with images
    d = tmp_path / "test_images.docx"
    import zipfile
    with zipfile.ZipFile(d, 'w') as z:
        z.writestr('word/media/image1.png', 'content')
    
    assert has_images_docx(d) == True

# Mocking OpenAI for process_image_with_ai would be ideal here
# but for now we just test the helper functions.
