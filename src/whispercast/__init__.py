"""WhisperCast: Unified Speech Intelligence Toolkit"""

# Expose the main CLI function for potential programmatic use
from .cli import main as run_whispercast_pipeline

# Expose core transcription and diarization functions if they are meant to be library functions
# For example, if you refactor them out of cli.py into core modules:
# from .core.transcription import transcribe_audio_local, transcribe_audio_openai_api
# from .core.diarization import diarize_audio_pyannote
# from .core.processing import process_transcription_to_dataframe

# Expose configuration variables or objects if useful
from .config import TRANSCRIPTION_MODE, OPENAI_API_MODEL, LOCAL_MODEL_ID, LOCAL_MODEL_DEVICE, HUGGING_FACE_ACCESS_TOKEN, FFMPEG_EXECUTABLE_PATH

__version__ = "0.1.0"  # Keep in sync with pyproject.toml

# You can also define __all__ to specify what `from whispercast import *` imports
# __all__ = ["run_whispercast_pipeline", "__version__"]
