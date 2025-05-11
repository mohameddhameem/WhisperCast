import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import openai
from dotenv import load_dotenv

# Was: from .. import config # Not strictly needed as config values are passed as arguments

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
            return_timestamps="segment",
            torch_dtype=torch_dtype,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1
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
                timestamp_granularities=["segment", "word"]
            )
        print("INFO: OpenAI API transcription complete.")
        return transcript
    except Exception as e:
        print(f"ERROR: OpenAI API transcription failed: {e}")
        return None
