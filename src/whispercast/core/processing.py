import pandas as pd

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
        if hasattr(transcription_response, 'segments') and transcription_response.segments:
            for i, segment in enumerate(transcription_response.segments):
                text = segment.text.strip() if hasattr(segment, 'text') and segment.text else ""
                start_time = segment.start if hasattr(segment, 'start') else None
                end_time = segment.end if hasattr(segment, 'end') else None

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
        elif hasattr(transcription_response, 'text') and transcription_response.text:
            print("WARNING: 'segments' not found in OpenAI API response. Using full text if available.")
            speaker_label = "Segment_1"
            if diarization_segments:
                 if diarization_segments: # Check if not empty
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
