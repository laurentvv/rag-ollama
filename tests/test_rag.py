import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import ollama
from src.rag_ollama.config import RAGConfig
from src.rag_ollama.rag import update_vector_db_incrementally

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

@patch('src.rag_ollama.rag.get_file_hash')
@patch('langchain_community.vectorstores.Chroma')
def test_incremental_indexing_new_file(MockChroma, mock_get_hash, config):
    """Tests that a new file is correctly added to the database."""
    # Setup
    mock_db = MockChroma.return_value
    mock_db.get.return_value = {'ids': []} # Simulate file not in DB for both get calls
    mock_get_hash.return_value = "new_hash_123"

    new_file = config.processed_dir / "new_doc.md"
    new_file.write_text("This is a new document.")

    # Execute
    update_vector_db_incrementally(mock_db, config)

    # Verify
    expected_get_calls = [
        call(where={'source': str(new_file), 'hash': 'new_hash_123'}),
        call(where={'source': str(new_file)})
    ]
    mock_db.get.assert_has_calls(expected_get_calls, any_order=False)
    mock_db.add_documents.assert_called_once()
    mock_db.delete.assert_not_called()

@patch('src.rag_ollama.rag.get_file_hash')
@patch('langchain_community.vectorstores.Chroma')
def test_incremental_indexing_unchanged_file(MockChroma, mock_get_hash, config):
    """Tests that an unchanged file is skipped."""
    # Setup
    mock_db = MockChroma.return_value
    mock_db.get.return_value = {'ids': ['some_id']}
    mock_get_hash.return_value = "existing_hash_456"

    existing_file = config.processed_dir / "existing_doc.md"
    existing_file.write_text("This is an existing document.")

    # Execute
    update_vector_db_incrementally(mock_db, config)

    # Verify
    mock_db.get.assert_called_once_with(where={"source": str(existing_file), "hash": "existing_hash_456"})
    mock_db.add_documents.assert_not_called()
    mock_db.delete.assert_not_called()

@patch('src.rag_ollama.rag.get_file_hash')
@patch('langchain_community.vectorstores.Chroma')
def test_incremental_indexing_modified_file(MockChroma, mock_get_hash, config):
    """Tests that a modified file is updated (deleted then added)."""
    # Setup
    mock_db = MockChroma.return_value
    mock_db.get.side_effect = [{'ids': []}, {'ids': ['old_id_123']}]
    mock_get_hash.return_value = "modified_hash_789"

    modified_file = config.processed_dir / "modified_doc.md"
    modified_file.write_text("This document has been modified.")

    # Execute
    update_vector_db_incrementally(mock_db, config)

    # Verify
    assert mock_db.get.call_count == 2
    mock_db.delete.assert_called_once_with(ids=['old_id_123'])
    mock_db.add_documents.assert_called_once()
