"""Configuration settings for the WhisperCast application.

This module centralizes all configuration parameters for WhisperCast,
making it easy to adjust settings for transcription modes, model identifiers,
API keys (though actual keys are best managed via .env or environment variables),
and external tool paths like FFmpeg.

Key Configuration Variables:
- `TRANSCRIPTION_MODE`: Determines whether to use 'local' or 'openai_api' for transcription.
- `LOCAL_MODEL_ID`: Hugging Face model ID for local transcription.
- `LOCAL_MODEL_DEVICE`: Device for local model inference (e.g., "cpu", "cuda").
- `OPENAI_API_MODEL`: Model name for OpenAI API transcription (e.g., "whisper-1").
- `FFMPEG_EXECUTABLE_PATH`: Optional path to FFmpeg binaries.
- `HUGGING_FACE_ACCESS_TOKEN`: Optional Hugging Face token, primarily for `pyannote.audio`.

Note: Sensitive information like API keys should ideally be loaded from
environment variables (e.g., via a `.env` file managed by `python-dotenv`)
rather than being hardcoded here, even if set to `None` as a placeholder.
The application attempts to load these from the environment first.
"""

# Configuration for the speech processing program

# Choose the transcription mode: 'local' or 'openai_api'
TRANSCRIPTION_MODE = 'local'  # or 'openai_api' local

# --- Local Model Configuration ---
# Specify the Hugging Face model ID for local transcription
# Example: "openai/whisper-large-v3", "openai/whisper-base", etc.
LOCAL_MODEL_ID = "openai/whisper-large-v3"
# Specify the device for local model: "cpu", "cuda" (if GPU is available and configured)
LOCAL_MODEL_DEVICE = "cpu"
# LOCAL_MODEL_DEVICE = "cuda"  # Uncomment this line if using GPU

# --- OpenAI API Configuration ---
# Specify the OpenAI API model name
OPENAI_API_MODEL = "whisper-1"

# --- FFmpeg Configuration ---
# Optional: Specify the path to the FFmpeg bin directory if it's not in the system PATH
# Set to None or an empty string if FFmpeg is in PATH or not needed.
# Example: FFMPEG_EXECUTABLE_PATH = r"C:\path\to\ffmpeg\bin"
FFMPEG_EXECUTABLE_PATH = r"C:\Learning\ffmpeg\bin"

# --- Hugging Face Configuration ---
# Optional: Set your Hugging Face access token here if you prefer it over an environment variable.
# The script will prioritize the HUGGING_FACE_ACCESS_TOKEN environment variable if set.
# Create a token at https://huggingface.co/settings/tokens
HUGGING_FACE_ACCESS_TOKEN = None # Or "YOUR_HUGGING_FACE_TOKEN_HERE" as an override
