"""Handles audio transcription using local models or the OpenAI API.

This module provides functions to transcribe audio files. It supports:
- Local transcription using Hugging Face Transformers (e.g., Whisper models).
- Transcription via the OpenAI API (Whisper).

The main entry point is the `transcribe_audio` function, which acts as a dispatcher
based on the selected mode.
"""

import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import openai
from dotenv import load_dotenv

# Was: from .. import config # Not strictly needed as config values are passed as arguments

def transcribe_audio_local(audio_file_path, model_id, device):
    """
    Transcribes an audio file using a locally loaded Hugging Face Whisper model.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        model_id (str): The Hugging Face model identifier (e.g., "openai/whisper-large-v3").
        device (str): The device to run the model on (e.g., "cpu", "cuda").

    Returns:
        dict or None: The transcription result from the Hugging Face pipeline,
                      containing text and timestamped segments/chunks, or None if transcription fails.
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
            return_timestamps="segment", # Changed from "word" to "segment" for local, can be "word" if model supports well
            torch_dtype=torch_dtype,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1 # Corrected device mapping for pipeline
        )
        print(f"INFO: Starting transcription for {audio_file_path}...")
        result = pipe(audio_file_path)
        print("INFO: Local transcription complete.")
        return result
    except Exception as e:
        print(f"ERROR: Local transcription failed: {e}")
        return None

def transcribe_audio_openai_api(audio_file_path, model_name):
    """
    Transcribes an audio file using the OpenAI Whisper API.

    Requires the `OPENAI_API_KEY` environment variable to be set.
    This function requests `verbose_json` output with `segment` and `word` level
    timestamp granularities. The returned object will contain these details if
    provided by the API.

    Args:
        audio_file_path (str): The path to the audio file (e.g., .wav, .mp3).
        model_name (str): The OpenAI Whisper model to use (e.g., "whisper-1").

    Returns:
        openai.types.audio.Transcription or None: The transcription object from the OpenAI API,
                                                   containing text, segments, and words with their
                                                   respective timestamps, or None if transcription
                                                   fails or API key is missing.
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
                response_format="verbose_json", # Ensures segments and words if available
                timestamp_granularities=["segment", "word"] # Request both
            )
        print("INFO: OpenAI API transcription complete.")
        # The 'transcript' object itself is what we need, it's not result.transcript
        return transcript # Return the direct response object
    except Exception as e:
        print(f"ERROR: OpenAI API transcription failed: {e}")
        return None

def transcribe_audio(audio_file_path, mode, local_model_id=None, local_model_device=None, openai_api_model=None):
    """
    Unified function to transcribe audio using either a local model or the OpenAI API.

    This function dispatches to `transcribe_audio_local` or `transcribe_audio_openai_api`
    based on the `mode` argument.

    Args:
        audio_file_path (str): Path to the input audio file.
        mode (str): Transcription mode, either 'local' or 'openai_api'.
        local_model_id (str, optional): Hugging Face model ID for local mode.
                                        Required if mode is 'local'. Defaults to None.
        local_model_device (str, optional): Device for local model (e.g., "cpu", "cuda").
                                           Required if mode is 'local'. Defaults to None.
        openai_api_model (str, optional): OpenAI API model name (e.g., "whisper-1").
                                         Required if mode is 'openai_api'. Defaults to None.

    Returns:
        dict or openai.types.audio.Transcription or None:
            The transcription result. Type depends on the mode used.
            Returns None if transcription fails or parameters are invalid.
    """
    if mode == 'local':
        if not local_model_id or not local_model_device:
            print("ERROR: local_model_id and local_model_device must be provided for local mode.")
            return None 
        return transcribe_audio_local(audio_file_path, local_model_id, local_model_device)
    elif mode == 'openai_api':
        if not openai_api_model:
            print("ERROR: openai_api_model must be provided for openai_api mode.")
            return None
        return transcribe_audio_openai_api(audio_file_path, openai_api_model)
    else:
        print(f"ERROR: Invalid transcription mode: {mode}. Choose 'local' or 'openai_api'.")
        return None
