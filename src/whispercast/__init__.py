"""WhisperCast: Unified Speech Intelligence Toolkit.

This package provides tools for audio transcription and speaker diarization,
leveraging both local models via Hugging Face Transformers and remote APIs
like OpenAI Whisper.

Core functionalities include:
- `transcribe_audio`: Transcribes an audio file.
- `perform_diarization`: Performs speaker diarization on an audio file.
- `process_transcription_and_diarization`: Combines transcription and diarization
  results into a structured format.

The package can also be used as a command-line tool via the `whispercast` script.
"""

# Expose configuration variables or objects if useful
# from .config import TRANSCRIPTION_MODE, OPENAI_API_MODEL, LOCAL_MODEL_ID, LOCAL_MODEL_DEVICE, HUGGING_FACE_ACCESS_TOKEN, FFMPEG_EXECUTABLE_PATH
# Commented out as these are not typically part of a library's public API surface directly from __init__
# They are used internally by the modules.

__version__ = "0.1.0"  # Keep in sync with pyproject.toml

# Option 1: Expose core functionalities directly
from .core import transcribe_audio, perform_diarization, process_transcription_and_diarization

__all__ = [
    "transcribe_audio",
    "perform_diarization",
    "process_transcription_and_diarization",
    "__version__",
]
