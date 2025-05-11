# Tests for the transcription module
import pytest
from whispercast.core.transcription import transcribe_audio
import importlib.resources
import os

def test_transcribe_audio_sample():
    """Test transcription with the packaged sample audio."""
    try:
        # Updated to use importlib.resources.files() and as_file()
        audio_path_ref = importlib.resources.files('whispercast.sample_data').joinpath('harvard.wav')
        with importlib.resources.as_file(audio_path_ref) as audio_path:
            audio_path_str = str(audio_path)
            # Corrected mode to 'openai_api' (underscore)
            # Also, pass the openai_api_model argument
            result = transcribe_audio(
                audio_file_path=audio_path_str,
                mode="openai_api", 
                openai_api_model="whisper-1" # Assuming default model for test
            )
            assert result is not None, "Transcription result should not be None"
            # Assuming OpenAI API response structure (verbose_json)
            assert hasattr(result, 'segments'), "Result should have 'segments' attribute"
            assert result.segments is not None, "Segments attribute should not be None"
            assert len(result.segments) > 0, "Segments list should not be empty"
            assert hasattr(result.segments[0], 'text'), "First segment should have 'text' attribute"
            assert hasattr(result, 'text') and result.text is not None, "Result should have a 'text' attribute that is not None"
            assert len(result.text.strip()) > 0, "Full transcription text should not be empty"
            print(f"Transcription result text: {result.text[:100]}...") # Log first 100 chars
    except FileNotFoundError:
        pytest.fail("The sample audio file 'harvard.wav' was not found in package data.")
    except Exception as e:
        pytest.fail(f"Transcription failed with an unexpected error: {e}")

# TODO: Add more tests for transcribe_audio, e.g. different models, invalid inputs
