"""Handles speaker diarization using pyannote.audio.

This module provides the `perform_diarization` function to identify speaker segments
in an audio file. It relies on the `pyannote.audio` library and may require a
Hugging Face access token for certain pre-trained pipelines.
"""

from pyannote.audio import Pipeline
# Was: from .. import config # Not strictly needed as config values are passed as arguments

def perform_diarization(audio_file_path, hf_token, pyannote_available_flag):
    """
    Performs speaker diarization using the `pyannote.audio` library.

    This function identifies speaker turns in an audio file and returns a list
    of segments, each with a start time, end time, and speaker label.

    Args:
        audio_file_path (str): Path to the input audio file.
        hf_token (str or None): Hugging Face access token. Required for some
                                `pyannote.audio` pipelines, especially gated models.
                                Can be None if the pipeline does not require authentication.
        pyannote_available_flag (bool): A flag indicating if `pyannote.audio` was successfully
                                      imported by the caller. If False, diarization is skipped.

    Returns:
        list[dict] or None: A list of dictionaries, where each dictionary represents a speaker
                            segment and contains 'start' (float), 'end' (float), and
                            'speaker' (str) keys. Returns None if diarization fails or
                            `pyannote.audio` is unavailable.
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
