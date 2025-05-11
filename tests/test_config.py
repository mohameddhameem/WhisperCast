# Tests for the config module
import pytest
from whispercast.config import OPENAI_API_MODEL, HUGGING_FACE_ACCESS_TOKEN # Corrected import names

# TODO: Add tests for config loading, perhaps with environment variable mocking
def test_api_keys_loaded_or_none():
    """Test that API keys are either strings or None."""
    # OPENAI_API_KEY is not directly in config, OPENAI_API_MODEL is.
    # If you intend to test the actual key, you'd need to load .env or mock os.environ
    assert isinstance(OPENAI_API_MODEL, str) # OPENAI_API_MODEL should always be a string
    assert isinstance(HUGGING_FACE_ACCESS_TOKEN, (str, type(None)))
