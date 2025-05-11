"""
Core functionalities for the WhisperCast package.

This package groups the main processing units of WhisperCast, including
audio transcription, speaker diarization, and the combination of their outputs.

Functions exposed:
- `transcribe_audio`: Performs audio transcription.
- `perform_diarization`: Conducts speaker diarization.
- `process_transcription_and_diarization`: Merges and formats transcription and diarization data.
- `save_output_to_csv`: Saves the processed data to a CSV file (Note: This was in the previous __all__ but the function itself is not defined in this file or imported. It seems to be part of processing.py but not explicitly exported here. Let's assume it should be if it was in __all__ before, or remove it if not intended to be public API of this sub-package directly).
"""
from .transcription import transcribe_audio
from .diarization import perform_diarization
from .processing import process_transcription_and_diarization, save_output_to_csv

__all__ = [
    "transcribe_audio",
    "perform_diarization",
    "process_transcription_and_diarization",
    "save_output_to_csv",
]
