from pyannote.audio import Pipeline
# Was: from .. import config # Not strictly needed as config values are passed as arguments

def diarize_audio_pyannote(audio_file_path, hf_token, pyannote_available_flag):
    """
    Performs speaker diarization using pyannote.audio.
    Returns a list of speaker segments with start, end, and speaker label.
    """
    if not pyannote_available_flag: # Check the flag passed from cli.py
        print("ERROR: pyannote.audio is not available (checked by caller). Cannot perform diarization.")
        return None

    print(f"INFO: Starting speaker diarization for {audio_file_path}...")
    try:
        # The pyannote_available_flag check is primarily done in cli.py before calling.
        # This function assumes Pipeline can be imported if pyannote_available_flag was True.
        if not hf_token or hf_token == "YOUR_HUGGING_FACE_TOKEN_HERE":
            print("WARNING: Hugging Face access token not provided or is a placeholder. Diarization model download might fail if it's a gated model.")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
        else:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
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
    except ImportError:
        # This catch is a fallback, primary check is via pyannote_available_flag
        print("ERROR: Failed to import pyannote.audio within diarization function. Ensure it's installed.")
        return None
    except Exception as e:
        print(f"ERROR: Speaker diarization failed: {e}")
        return None
