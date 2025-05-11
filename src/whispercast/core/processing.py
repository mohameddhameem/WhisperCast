"""Processes transcription and diarization results into a structured format.

This module provides functions to combine the raw outputs from transcription
(from local models or OpenAI API) and speaker diarization (from pyannote.audio)
into a pandas DataFrame. This structured format typically includes columns for
speaker, start time, end time, and the transcribed text segment.
"""

import pandas as pd

def process_transcription_and_diarization(transcription_response, mode, diarization_segments=None):
    """
    Processes transcription results and optional speaker diarization data into a pandas DataFrame.

    Handles different output structures from local Hugging Face pipelines (expected to have 'chunks')
    and the OpenAI API (expected to have 'segments' or 'words' in a `verbose_json` response).
    If diarization data is provided, it attempts to assign speaker labels to transcription segments
    based on temporal overlap.

    Args:
        transcription_response (dict or openai.types.audio.Transcription or None):
            The raw output from a transcription function.
            - For 'local' mode: Expected to be a dict with a 'chunks' key, where each chunk
              has 'text' and 'timestamp' (a tuple of start, end times).
            - For 'openai_api' mode: Expected to be an OpenAI `Transcription` object, typically
              from a `verbose_json` response, with a `segments` attribute.
            Can be None if transcription failed.
        mode (str): The transcription mode used ('local' or 'openai_api'). This dictates how
                    `transcription_response` is parsed.
        diarization_segments (list[dict], optional): A list of speaker segments from diarization.
            Each dict should have 'start' (float), 'end' (float), and 'speaker' (str) keys.
            Defaults to None if diarization was not performed or failed.

    Returns:
        pandas.DataFrame: A DataFrame with columns ['Speaker', 'Start Time', 'End Time', 'Conversation'].
                          Returns an empty DataFrame if processing fails or no data is generated.
    """
    rows = []
    print("INFO: Processing transcription response...")

    if not transcription_response:
        print("WARNING: No transcription response to process.")
        return pd.DataFrame(columns=['Speaker', 'Start Time', 'End Time', 'Conversation'])

    # Helper function to find speaker for a given time range
    def get_speaker_for_segment(segment_start, segment_end, diarization_data):
        """Assigns a speaker label to a transcription segment based on diarization data.

        Identifies the speaker who was most active during the time interval of the
        transcription segment.

        Args:
            segment_start (float): Start time of the transcription segment.
            segment_end (float): End time of the transcription segment.
            diarization_data (list[dict]): List of speaker turns from diarization.

        Returns:
            str or None: The label of the most active speaker in the segment's timeframe,
                         or "Unknown_Speaker" if no speaker overlaps significantly, or None
                         if diarization_data is empty.
        """
        if not diarization_data:
            return None # No diarization data available

        active_speakers = {}
        for turn in diarization_data:
            overlap_start = max(segment_start, turn['start'])
            overlap_end = min(segment_end, turn['end'])
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                if turn['speaker'] not in active_speakers:
                    active_speakers[turn['speaker']] = 0
                active_speakers[turn['speaker']] += overlap_duration

        if not active_speakers:
            return "Unknown_Speaker"

        return max(active_speakers, key=active_speakers.get)

    if mode == 'local':
        if 'chunks' in transcription_response:
            for i, chunk in enumerate(transcription_response['chunks']):
                text = chunk['text'].strip()
                start_time, end_time = chunk['timestamp']

                speaker_label = f"Segment_{i+1}"
                if diarization_segments and start_time is not None and end_time is not None:
                    speaker_label = get_speaker_for_segment(start_time, end_time, diarization_segments) or speaker_label

                if text:
                    rows.append({
                        'Speaker': speaker_label,
                        'Start Time': start_time if start_time is not None else "N/A",
                        'End Time': end_time if end_time is not None else "N/A",
                        'Conversation': text
                    })
        elif 'text' in transcription_response and transcription_response['text']:
            print("WARNING: 'chunks' not found in local transcription response. Using full text if available.")
            speaker_label = "Segment_1"
            if diarization_segments:
                if diarization_segments: # Check if not empty
                    speaker_label = diarization_segments[0]['speaker']
            rows.append({
                'Speaker': speaker_label,
                'Start Time': "N/A",
                'End Time': "N/A",
                'Conversation': transcription_response['text'].strip()
            })

    elif mode == 'openai_api':
        if diarization_segments:
            # Process with diarization: segment-level, assign speakers
            if hasattr(transcription_response, 'segments') and transcription_response.segments:
                for i, segment in enumerate(transcription_response.segments):
                    text = segment.text.strip() if hasattr(segment, 'text') and segment.text else ""
                    start_time = segment.start if hasattr(segment, 'start') else None
                    end_time = segment.end if hasattr(segment, 'end') else None

                    speaker_label = f"Segment_{i+1}" # Default if no speaker found
                    if start_time is not None and end_time is not None:
                        speaker_label = get_speaker_for_segment(start_time, end_time, diarization_segments) or speaker_label

                    if text:
                        rows.append({
                            'Speaker': speaker_label,
                            'Start Time': start_time if start_time is not None else "N/A",
                            'End Time': end_time if end_time is not None else "N/A",
                            'Conversation': text
                        })
            elif hasattr(transcription_response, 'text') and transcription_response.text:
                # Fallback for OpenAI if segments not present but diarization is on (less ideal)
                print("WARNING: 'segments' not found in OpenAI API response with diarization. Using full text.")
                speaker_label = "Segment_1"
                if diarization_segments: # Check if not empty
                     speaker_label = get_speaker_for_segment(0, transcription_response.duration if hasattr(transcription_response, 'duration') else float('inf'), diarization_segments) or speaker_label
                rows.append({
                    'Speaker': speaker_label,
                    'Start Time': 0 if hasattr(transcription_response, 'duration') else "N/A",
                    'End Time': transcription_response.duration if hasattr(transcription_response, 'duration') else "N/A",
                    'Conversation': transcription_response.text.strip()
                })
            else:
                print("WARNING: No processable segments or text in OpenAI API response with diarization.")
        else:
            # No diarization: try word-level, then segment-level
            if hasattr(transcription_response, 'words') and transcription_response.words:
                print("INFO: Processing word-level timestamps from OpenAI API response.")
                for word_info in transcription_response.words:
                    word_text = word_info.word.strip() if hasattr(word_info, 'word') and word_info.word else ""
                    start_time = word_info.start if hasattr(word_info, 'start') else None
                    end_time = word_info.end if hasattr(word_info, 'end') else None
                    if word_text:
                        rows.append({
                            'Speaker': 'N/A', # No speaker info for word-level without diarization
                            'Start Time': start_time if start_time is not None else "N/A",
                            'End Time': end_time if end_time is not None else "N/A",
                            'Conversation': word_text
                        })
            elif hasattr(transcription_response, 'segments') and transcription_response.segments:
                print("INFO: Word-level timestamps not found or empty. Processing segment-level timestamps from OpenAI API response (no diarization).")
                for i, segment in enumerate(transcription_response.segments):
                    text = segment.text.strip() if hasattr(segment, 'text') and segment.text else ""
                    start_time = segment.start if hasattr(segment, 'start') else None
                    end_time = segment.end if hasattr(segment, 'end') else None
                    if text:
                        rows.append({
                            'Speaker': 'N/A', # No speaker info
                            'Start Time': start_time if start_time is not None else "N/A",
                            'End Time': end_time if end_time is not None else "N/A",
                            'Conversation': text
                        })
            elif hasattr(transcription_response, 'text') and transcription_response.text:
                print("WARNING: Neither words nor segments found in OpenAI API response (no diarization). Using full text.")
                rows.append({
                    'Speaker': 'N/A',
                    'Start Time': 0 if hasattr(transcription_response, 'duration') else "N/A",
                    'End Time': transcription_response.duration if hasattr(transcription_response, 'duration') else "N/A",
                    'Conversation': transcription_response.text.strip()
                })
            else:
                print("WARNING: No processable words, segments, or text in OpenAI API response (no diarization).")

    if not rows:
        print("WARNING: No data rows were generated from the transcription.")
        return pd.DataFrame(columns=['Speaker', 'Start Time', 'End Time', 'Conversation'])

    df = pd.DataFrame(rows)
    print(f"INFO: DataFrame created with {len(df)} rows.")
    return df
