import pytest
from unittest.mock import MagicMock

from src.query_processor import HSNQueryProcessor

@pytest.fixture
def mock_rag_system():
    """Creates a mock RAG system."""
    return MagicMock(spec=HSNQueryProcessor)

def test_parse_natural_language_intent_detection(mock_rag_system):
    """Tests the intent classification logic in isolation."""
    processor = HSNQueryProcessor(mock_rag_system)
    
    # Test classification intent
    result = processor._parse_natural_language("hsn for rubber")
    assert result["intent"] == "classification"

    # Test summarization intent
    result = processor._parse_natural_language("what are the types of rubber?")
    assert result["intent"] == "summarization"

    # Test selection intent
    result = processor._parse_natural_language("option 2")
    assert result["intent"] == "selection"
    result = processor._parse_natural_language("choose the first one")
    assert result["intent"] == "selection"