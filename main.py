import argparse
import os
import config  # Your configuration file

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
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime # Added import

# Flags to track availability of local mode dependencies
torch_available = False
transformers_components_available = False
pyannote_available = False # Flag for pyannote.audio

# Attempt to import local model libraries, fail gracefully if not in local mode or not installed
try:
    import torch
    torch_available = True
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    transformers_components_available = True
except ImportError as e:
    print(f"DEBUG: An ImportError occurred during initial imports: {e}")
    if config.TRANSCRIPTION_MODE == 'local':
        print("INFO: Failed to import 'torch' and/or 'transformers' components. Local mode may not work. Please ensure they are installed correctly in your environment.")
    # No need to raise error here if not in local mode, main() will handle it

# Attempt to import pyannote.audio
try:
    from pyannote.audio import Pipeline
    pyannote_available = True
except ImportError:
    print("INFO: 'pyannote.audio' library not found. Speaker diarization will not be available. Install with: pip install pyannote.audio")

# Attempt to import OpenAI library, fail gracefully if not in openai_api mode or not installed
try:
    import openai
except ImportError:
    if config.TRANSCRIPTION_MODE == 'openai_api':
        print("INFO: 'openai' library not found. Please install it if using OpenAI API mode: pip install openai")

def transcribe_audio_local(audio_file_path, model_id, device):
    """
    Transcribes audio using a local Hugging Face Whisper model.
    """
    print(f"INFO: Loading local model '{model_id}' on device '{device}'...")
    try:
        # Determine torch_dtype based on device/CUDA availability
        if device == "cuda":
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                print("INFO: Using torch.float16 for CUDA.")
            else:
                print("WARNING: Device set to 'cuda' but CUDA is not available. Falling back to CPU and float32.")
                device = "cpu" # Fallback to CPU
                torch_dtype = torch.float32
        else: # CPU
            torch_dtype = torch.float32
            print("INFO: Using torch.float32 for CPU.")

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device) # Send model to the specified device

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128, # Adjust as needed
            # chunk_length_s=30, # Default is 30s for long-form transcription
            # batch_size=16, # Adjust based on your hardware
            return_timestamps="segment", # Changed from "word" to "segment"
            torch_dtype=torch_dtype,  # Pass determined dtype to pipeline
            device=0 if device == "cuda" and torch.cuda.is_available() else -1 # device=0 for cuda, -1 for cpu
        )
        print(f"INFO: Starting transcription for {audio_file_path}...")
        # The pipeline expects the audio file path directly or loaded audio data
        result = pipe(audio_file_path)
        print("INFO: Local transcription complete.")
        return result
    except Exception as e:
        print(f"ERROR: Local transcription failed: {e}")
        # You might want to print more detailed error info or traceback here
        # import traceback
        # traceback.print_exc()
        return None

def transcribe_audio_openai_api(audio_file_path, model_name):
    """
    Transcribes audio using the OpenAI Whisper API.
    """
    print(f"INFO: Transcribing with OpenAI API model '{model_name}'...")
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in .env file or environment variables.")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                response_format="verbose_json", 
                timestamp_granularities=["segment", "word"] # Requesting both segment and word timestamps
            )
        print("INFO: OpenAI API transcription complete.")
        return transcript # This will be an OpenAIObject, not a direct dict in some SDK versions
    except Exception as e:
        print(f"ERROR: OpenAI API transcription failed: {e}")
        return None

# --- Speaker Diarization Function ---
def diarize_audio_pyannote(audio_file_path, hf_token):
    """
    Performs speaker diarization using pyannote.audio.
    Returns a list of speaker segments with start, end, and speaker label.
    """
    if not pyannote_available:
        print("ERROR: pyannote.audio is not available. Cannot perform diarization.")
        return None
    
    print(f"INFO: Starting speaker diarization for {audio_file_path}...")
    try:
        if not hf_token or hf_token == "YOUR_HUGGING_FACE_TOKEN_HERE":
            print("WARNING: Hugging Face access token not provided or is a placeholder in config.py or .env. Diarization model download might fail if it's a gated model.")
            # Attempt to use pipeline without token, might work for non-gated or cached models
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-pytorch")
        else:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-pytorch", 
                use_auth_token=hf_token
            )
        
        diarization = diarization_pipeline(audio_file_path)
        
        processed_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            processed_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        print(f"INFO: Speaker diarization complete. Found {len(processed_segments)} speaker turns.")
        return processed_segments
    except Exception as e:
        print(f"ERROR: Speaker diarization failed: {e}")
        # import traceback
        # traceback.print_exc()
        return None

def process_transcription_to_dataframe(transcription_response, mode, diarization_segments=None):
    """
    Processes the transcription response and optional diarization info into a Pandas DataFrame.
    """
    rows = []
    print("INFO: Processing transcription response...")

    if not transcription_response:
        print("WARNING: No transcription response to process.")
        return pd.DataFrame(columns=['Speaker', 'Start Time', 'End Time', 'Conversation'])

    # Helper function to find speaker for a given time range
    def get_speaker_for_segment(segment_start, segment_end, diarization_data):
        if not diarization_data:
            return None # No diarization data available
        
        active_speakers = {}
        for turn in diarization_data:
            # Check for overlap
            overlap_start = max(segment_start, turn['start'])
            overlap_end = min(segment_end, turn['end'])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                if turn['speaker'] not in active_speakers:
                    active_speakers[turn['speaker']] = 0
                active_speakers[turn['speaker']] += overlap_duration
        
        if not active_speakers:
            return "Unknown_Speaker" # No speaker overlapped with this segment
        
        # Return speaker with maximum overlap
        return max(active_speakers, key=active_speakers.get)

    if mode == 'local':
        # The Hugging Face pipeline with return_timestamps="word" gives a list of dicts in 'chunks'
        # Each chunk has 'text' and 'timestamp' (tuple: start, end)
        if 'chunks' in transcription_response: # Check if 'chunks' key exists
            for i, chunk in enumerate(transcription_response['chunks']):
                text = chunk['text'].strip()
                start_time, end_time = chunk['timestamp']
                
                speaker_label = f"Segment_{i+1}" # Default if no diarization
                if diarization_segments and start_time is not None and end_time is not None:
                    speaker_label = get_speaker_for_segment(start_time, end_time, diarization_segments) or speaker_label

                if text:
                    rows.append({
                        'Speaker': speaker_label,
                        'Start Time': start_time if start_time is not None else "N/A",
                        'End Time': end_time if end_time is not None else "N/A",
                        'Conversation': text
                    })
        else: # Fallback if 'chunks' is not present, try to use the main 'text'
             print("WARNING: 'chunks' not found in local transcription response. Using full text if available.")
             if 'text' in transcription_response and transcription_response['text']:
                 speaker_label = "Segment_1"
                 if diarization_segments: # Basic check, won't be accurate without timestamps
                     # Attempt to assign the first speaker if diarization data exists, very rough
                     if diarization_segments:
                         speaker_label = diarization_segments[0]['speaker']
                 rows.append({
                        'Speaker': speaker_label,
                        'Start Time': "N/A", # Timestamps might not be available at this level
                        'End Time': "N/A",
                        'Conversation': transcription_response['text'].strip()
                    })


    elif mode == 'openai_api':
        # OpenAI API response (verbose_json) has 'segments'
        # Each segment is an object with attributes like 'start', 'end', 'text'
        if hasattr(transcription_response, 'segments') and transcription_response.segments:
            for i, segment in enumerate(transcription_response.segments):
                text = segment.text.strip() if hasattr(segment, 'text') and segment.text else ""
                start_time = segment.start if hasattr(segment, 'start') else None
                end_time = segment.end if hasattr(segment, 'end') else None

                speaker_label = f"Segment_{i+1}" # Default if no diarization
                if diarization_segments and start_time is not None and end_time is not None:
                    speaker_label = get_speaker_for_segment(start_time, end_time, diarization_segments) or speaker_label
                
                if text:
                    rows.append({
                        'Speaker': speaker_label,
                        'Start Time': start_time if start_time is not None else "N/A",
                        'End Time': end_time if end_time is not None else "N/A",
                        'Conversation': text
                    })
        # ... (rest of openai_api mode fallback as before)
        elif hasattr(transcription_response, 'text') and transcription_response.text: 
            print("WARNING: 'segments' not found in OpenAI API response. Using full text if available.")
            speaker_label = "Segment_1"
            if diarization_segments:
                 if diarization_segments:
                     speaker_label = diarization_segments[0]['speaker']
            rows.append({
                'Speaker': speaker_label,
                'Start Time': "N/A",
                'End Time': "N/A",
                'Conversation': transcription_response.text.strip()
            })
        else:
            print("WARNING: No processable segments or text in OpenAI API response.")

    if not rows:
        print("WARNING: No data rows were generated from the transcription.")
        return pd.DataFrame(columns=['Speaker', 'Start Time', 'End Time', 'Conversation'])
        
    df = pd.DataFrame(rows)
    print(f"INFO: DataFrame created with {len(df)} rows.")
    return df

def main():
    # Define default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjusted default_audio_path based on workspace structure: speech/harvard.wav/harvard.wav
    default_audio_path = os.path.join(script_dir, "harvard.wav", "harvard.wav") 
    
    # Generate default CSV filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%I-%M-%S_%p") # e.g., 2025-05-11_03-45-20_PM
    default_csv_filename = f"output_{timestamp}.csv"
    default_output_csv_path = os.path.join(script_dir, "output", default_csv_filename)

    parser = argparse.ArgumentParser(description="Transcribe audio and output to CSV.")
    parser.add_argument("audio_file", nargs='?', default=default_audio_path, 
                        help=f"Path to the input .wav audio file (default: {default_audio_path}).")
    parser.add_argument("output_csv", nargs='?', default=default_output_csv_path,
                        help=f"Path to save the output CSV file (default: {default_output_csv_path}).")
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
        # if 'transformers' not in globals() or 'torch' not in globals(): # Old problematic check
        if not torch_available or not transformers_components_available:
            print("ERROR: Local mode selected but required libraries (torch and/or transformers components) failed to import. Please check installation and import statements.")
            if not torch_available:
                print("DEBUG: 'torch' module was not successfully imported.")
            if not transformers_components_available:
                print("DEBUG: Components from 'transformers' (e.g., AutoModelForSpeechSeq2Seq) were not successfully imported.")
            return
        transcription_result = transcribe_audio_local(args.audio_file, config.LOCAL_MODEL_ID, config.LOCAL_MODEL_DEVICE)
    elif config.TRANSCRIPTION_MODE == 'openai_api':
        if 'openai' not in globals():
            print("ERROR: OpenAI API mode selected but 'openai' library is not available. Please install it.")
            return
        transcription_result = transcribe_audio_openai_api(args.audio_file, config.OPENAI_API_MODEL)
    else:
        print(f"ERROR: Invalid TRANSCRIPTION_MODE '{config.TRANSCRIPTION_MODE}' in config.py. Choose 'local' or 'openai_api'.")
        return

    # --- Speaker Diarization (Optional) ---
    diarization_result = None
    if pyannote_available: # Only run if pyannote was imported successfully
        hf_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN") or config.HUGGING_FACE_ACCESS_TOKEN
        if not hf_token or hf_token == "YOUR_HUGGING_FACE_TOKEN_HERE":
            print("INFO: HUGGING_FACE_ACCESS_TOKEN not found or is a placeholder. Diarization might fail for gated models.")
            # Proceeding without token, pyannote will try to use cached models or public ones.
        diarization_result = diarize_audio_pyannote(args.audio_file, hf_token)
    else:
        print("INFO: pyannote.audio not available, skipping speaker diarization.")

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
