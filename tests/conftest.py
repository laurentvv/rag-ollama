import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ollama():
    """Mocks the ollama.Client and its chat method."""
    with patch('ollama.Client') as mock_client:
        mock_chat = MagicMock()
        mock_chat.return_value = {'message': {'content': 'Mocked AI response.'}}
        mock_client.return_value.chat = mock_chat
        yield mock_client
