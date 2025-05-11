"""Command-Line Interface for WhisperCast.

This module provides the command-line entry point for the WhisperCast application.
It handles argument parsing, initializes the transcription and diarization processes
based on user input and configuration, and saves the output.

Supports transcription via local Hugging Face models or the OpenAI API, with
optional speaker diarization using pyannote.audio.

Usage:
  whispercast [audio_file] [output_csv] [--enable-diarization] [--version]
"""

import argparse
import os
from . import config # New relative import for the package
import importlib.resources
from . import __version__ # Import the version from __init__.py

# --- Prepend FFmpeg path to environment PATH ---
if config.FFMPEG_EXECUTABLE_PATH and os.path.isdir(config.FFMPEG_EXECUTABLE_PATH):
    print(f"INFO: Adding {config.FFMPEG_EXECUTABLE_PATH} to PATH for this session.")
    os.environ["PATH"] = config.FFMPEG_EXECUTABLE_PATH + os.pathsep + os.environ["PATH"]
elif config.FFMPEG_EXECUTABLE_PATH: # If path is set in config but not found
    print(f"WARNING: Specified FFmpeg directory {config.FFMPEG_EXECUTABLE_PATH} in config.py not found. Audio loading might fail if FFmpeg is not in system PATH.")
else: # If FFMPEG_EXECUTABLE_PATH is None or empty in config
    print("INFO: FFMPEG_EXECUTABLE_PATH not set in config.py. Assuming FFmpeg is in system PATH or not required.")
# --- End FFmpeg path modification ---

# Now import other modules that depend on config
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
# Import core functionalities
from .core.transcription import transcribe_audio # Unified function
from .core.diarization import perform_diarization # Renamed from diarize_audio_pyannote
from .core.processing import process_transcription_and_diarization # Renamed from process_transcription_to_dataframe

# Flags to track availability of local mode dependencies
torch_available = False
transformers_components_available = False
pyannote_available = False # Flag for pyannote.audio

# Attempt to import local model libraries, fail gracefully if not in local mode or not installed
try:
    import torch # Keep for torch_available check and config.LOCAL_MODEL_DEVICE logic
    torch_available = True
    # from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline # Moved to core.transcription
    # Check if the core components were intended to be checked here or if just torch is enough
    # For now, assuming core.transcription handles its own transformers imports.
    # If a global check for transformers is needed, it should be more specific.
    # We can assume transformers is available if torch is, for the purpose of this flag,
    # or rely on the more specific error handling within transcribe_audio_local.
    # To keep it simple, let's assume if torch is here, transformers *should* be too for local mode.
    # A more robust check might involve trying to import a specific class from transformers.
    # However, the actual import and use is in core.transcription.
    # Let's set transformers_components_available based on torch for now,
    # as the detailed check happens in transcribe_audio_local.
    if torch_available: # Simplified assumption
        transformers_components_available = True # This flag is checked before calling transcribe_audio_local
except ImportError as e:
    print(f"DEBUG: An ImportError occurred during initial imports (torch): {e}")
    if config.TRANSCRIPTION_MODE == 'local':
        print("INFO: Failed to import 'torch'. Local mode may not work. Please ensure it's installed correctly in your environment.")
    # transformers_components_available will remain False

# Attempt to import pyannote.audio
try:
    from pyannote.audio import Pipeline # Keep for pyannote_available check
    pyannote_available = True
except ImportError:
    print("INFO: 'pyannote.audio' library not found. Speaker diarization will not be available. Install with: pip install pyannote.audio")

# Attempt to import OpenAI library, fail gracefully if not in openai_api mode or not installed
try:
    import openai # Keep for \'openai\' in globals() check
except ImportError:
    if config.TRANSCRIPTION_MODE == 'openai_api':
        print("INFO: \'openai\' library not found. Please install it if using OpenAI API mode: pip install openai")

def main():
    """Main function for the WhisperCast CLI.

    Parses command-line arguments, sets up default paths, and orchestrates
    the audio transcription and optional diarization process.

    The process involves:
    1. Setting up paths for input audio and output CSV.
    2. Parsing CLI arguments for audio file, output file, and diarization flag.
    3. Performing transcription based on the mode specified in `config.py` (local or openai_api).
    4. Optionally performing speaker diarization if requested and `pyannote.audio` is available.
    5. Processing the results into a pandas DataFrame.
    6. Saving the DataFrame to a CSV file.
    """
    # Define default paths
    # script_dir = os.path.dirname(os.path.abspath(__file__)) # No longer needed for default_audio_path
    
    # Use importlib.resources to get the path to the sample audio file
    try:
        default_audio_path_ref = importlib.resources.files('whispercast.sample_data').joinpath('harvard.wav')
        # For Python < 3.9, importlib.resources.path was used.
        # For Python >= 3.9, files() returns a Traversable object.
        # To get a usable file system path, we enter its context if it's a temporary file,
        # or just use it directly if it's a real file path.
        # Given our setup, it should be a direct path within the package.
        with importlib.resources.as_file(default_audio_path_ref) as path:
            default_audio_path = str(path)
    except Exception as e:
        print(f"WARNING: Could not locate default audio file using importlib.resources: {e}")
        # Fallback or error, though this should ideally not happen if packaged correctly
        # For a CLI tool, a fallback might be to not have a default or point to a placeholder.
        # Let's set it to a non-existent placeholder to indicate an issue if it fails.
        default_audio_path = "default_sample_audio_not_found.wav"


    # Generate default CSV filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%I-%M-%S_%p") # e.g., 2025-05-11_03-45-20_PM
    default_csv_filename = f"output_{timestamp}.csv"
    
    # Default output path relative to the current working directory
    default_output_csv_path = os.path.join(os.getcwd(), "output", default_csv_filename)


    parser = argparse.ArgumentParser(description="Transcribe audio and output to CSV.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit."
    )
    parser.add_argument("audio_file", nargs='?', default=default_audio_path, 
                        help=f"Path to the input .wav audio file (default: {default_audio_path}).")
    parser.add_argument("output_csv", nargs='?', default=default_output_csv_path,
                        help=f"Path to save the output CSV file (default: {default_output_csv_path}).")
    parser.add_argument(
        "--enable-diarization",
        action="store_true", # Makes it a boolean flag, default is False
        help="Enable speaker diarization. Requires pyannote.audio and a Hugging Face token."
    )
    args = parser.parse_args()

    print(f"INFO: Using input audio file: {args.audio_file}")
    print(f"INFO: Using output CSV file: {args.output_csv}")

    if not os.path.exists(args.audio_file):
        print(f"ERROR: Audio file not found at {args.audio_file}")
        return

    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir) # Ensure output directory exists
        print(f"INFO: Created output directory: {output_dir}")

    # --- Transcription --- 
    transcription_result = None
    if config.TRANSCRIPTION_MODE == 'local':
        if not torch_available or not transformers_components_available: # transformers_components_available check might be redundant if transcribe_audio_local handles it robustly
            print("ERROR: Local mode selected but required libraries (torch and/or transformers components) failed to import or are not available. Please check installation and import statements.")
            if not torch_available:
                print("DEBUG: 'torch' module was not successfully imported.")
            if not transformers_components_available:
                print("DEBUG: Components from 'transformers' (e.g., AutoModelForSpeechSeq2Seq) were not successfully imported.")
            return
        # Corrected: Call the unified transcribe_audio function for local mode
        transcription_result = transcribe_audio(
            audio_file_path=args.audio_file,
            mode='local',
            local_model_id=config.LOCAL_MODEL_ID,
            local_model_device=config.LOCAL_MODEL_DEVICE
        )
    elif config.TRANSCRIPTION_MODE == 'openai_api':
        # Corrected: Call the unified transcribe_audio function for OpenAI API mode
        transcription_result = transcribe_audio(
            audio_file_path=args.audio_file,
            mode='openai_api',
            openai_api_model=config.OPENAI_API_MODEL
        )
    else:
        print(f"ERROR: Invalid TRANSCRIPTION_MODE '{config.TRANSCRIPTION_MODE}' in config.py. Choose 'local' or 'openai_api'.")
        return

    # --- Speaker Diarization (Optional) ---
    diarization_result = None
    if args.enable_diarization:
        if pyannote_available:
            print("INFO: Speaker diarization enabled by user.")
            hf_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN") or config.HUGGING_FACE_ACCESS_TOKEN
            if not hf_token or hf_token == "YOUR_HUGGING_FACE_TOKEN_HERE":
                print("INFO: HUGGING_FACE_ACCESS_TOKEN not found or is a placeholder. Diarization might fail for gated models.")
            # The pyannote_available flag is checked before calling.
            # The core diarization function also has its own checks.
            diarization_result = perform_diarization(args.audio_file, hf_token, pyannote_available) # Corrected: use perform_diarization
        else:
            print("WARNING: Speaker diarization was requested, but pyannote.audio is not available. Skipping diarization.")
    else:
        print("INFO: Speaker diarization not enabled. Skipping.")

    # --- Process and Save --- 
    if transcription_result:
        df = process_transcription_and_diarization(transcription_result, config.TRANSCRIPTION_MODE, diarization_result) # Corrected: use process_transcription_and_diarization
        if not df.empty:
            df.to_csv(args.output_csv, index=False, encoding='utf-8')
            print(f"INFO: Transcription saved to {args.output_csv}")
        else:
            print("WARNING: DataFrame is empty. CSV file not saved.")
    else:
        print("ERROR: Transcription failed. No output generated.")

if __name__ == "__main__":
    main()
