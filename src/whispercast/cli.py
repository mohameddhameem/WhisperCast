import argparse
import os
from . import config # New relative import for the package

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
from .core.transcription import transcribe_audio_local, transcribe_audio_openai_api
from .core.diarization import diarize_audio_pyannote
from .core.processing import process_transcription_to_dataframe

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
    # Define default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjusted default_audio_path to navigate up from src/whispercast
    default_audio_path = os.path.join(script_dir, "..", "..", "harvard.wav", "harvard.wav") 
    
    # Generate default CSV filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%I-%M-%S_%p") # e.g., 2025-05-11_03-45-20_PM
    default_csv_filename = f"output_{timestamp}.csv"
    # Adjusted default_output_csv_path to navigate up from src/whispercast
    default_output_csv_path = os.path.join(script_dir, "..", "..", "output", default_csv_filename)

    parser = argparse.ArgumentParser(description="Transcribe audio and output to CSV.")
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
        transcription_result = transcribe_audio_local(args.audio_file, config.LOCAL_MODEL_ID, config.LOCAL_MODEL_DEVICE)
    elif config.TRANSCRIPTION_MODE == 'openai_api':
        transcription_result = transcribe_audio_openai_api(args.audio_file, config.OPENAI_API_MODEL)
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
            diarization_result = diarize_audio_pyannote(args.audio_file, hf_token, pyannote_available)
        else:
            print("WARNING: Speaker diarization was requested, but pyannote.audio is not available. Skipping diarization.")
    else:
        print("INFO: Speaker diarization not enabled. Skipping.")

    # --- Process and Save --- 
    if transcription_result:
        df = process_transcription_to_dataframe(transcription_result, config.TRANSCRIPTION_MODE, diarization_result)
        if not df.empty:
            df.to_csv(args.output_csv, index=False, encoding='utf-8')
            print(f"INFO: Transcription saved to {args.output_csv}")
        else:
            print("WARNING: DataFrame is empty. CSV file not saved.")
    else:
        print("ERROR: Transcription failed. No output generated.")

if __name__ == "__main__":
    main()
