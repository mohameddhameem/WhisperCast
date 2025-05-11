# 🗣️ WhisperCast: Unified Speech Intelligence Toolkit

🔍 **Description:** WhisperCast is a unified speech intelligence toolkit powered by OpenAI and Hugging Face Whisper models. It enables high-quality transcription and speaker diarization from audio inputs. Ideal for multilingual voice processing, podcast indexing, meeting analytics, and more — all using open-source and API-based models.

---

## ✨ Features

*   **High-Quality Transcription**: Leverages state-of-the-art speech-to-text models.
    *   Choice between local Hugging Face Whisper models (e.g., `openai/whisper-large-v3`) for offline processing and privacy.
    *   Option to use the OpenAI Whisper API for potentially faster processing and different model variants.
*   **Speaker Diarization**: Identifies and attributes speech segments to different speakers using `pyannote.audio`.
*   **Timestamped Output**: Provides start and end times for each transcribed segment and speaker.
*   **Flexible Configuration**: Easily switch between transcription modes (local vs. OpenAI API) and configure model parameters via `config.py`.
*   **CSV Output**: Generates a structured CSV file with columns: `Speaker`, `Start Time`, `End Time`, `Conversation`.
*   **Handles Mixed Languages**: Designed to process audio with multiple languages (testing ongoing).
*   **(Planned) Translation**: Future support for translating transcribed text.

---

## 🚀 Tech Stack

*   **Python 3.x**
*   **Core Libraries**:
    *   [OpenAI API](https://platform.openai.com/docs/api-reference/audio): For API-based transcription.
    *   [Hugging Face Transformers](https://huggingface.co/docs/transformers/index): For local Whisper models.
    *   [Pyannote.audio](https://github.com/pyannote/pyannote-audio): For speaker diarization.
    *   [Pandas](https://pandas.pydata.org/): For data manipulation and CSV output.
    *   [Torch](https://pytorch.org/): Backend for Hugging Face Transformers and Pyannote.
*   **Audio Processing**:
    *   [FFmpeg](https://ffmpeg.org/): Required for audio loading and processing by local models.

---

## 📂 Project Structure

```
.
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── config.py           # Configuration for transcription mode, model IDs, API keys, etc.
├── main.py             # Main script to run the transcription and diarization pipeline
├── requirements.txt    # Python dependencies
├── .env.example        # Example for environment variables (OpenAI API Key, Hugging Face Token)
├── harvard.wav/        # Sample audio file directory (contains harvard.wav)
│   └── harvard.wav
├── output/             # Default directory for CSV outputs (ignored by Git)
└── README.md           # This file
```

---

## 🛠️ Setup and Installation

### Prerequisites

*   **Python**: Version 3.9 or higher recommended.
*   **FFmpeg**: Must be installed and accessible in your system's PATH, or its path specified in `config.py`.
    *   **Windows**: Download from [FFmpeg Official Site](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH.
    *   **macOS**: `brew install ffmpeg`
    *   **Linux**: `sudo apt update && sudo apt install ffmpeg`

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd WhisperCast # Or your project directory name
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   Windows (PowerShell/CMD): `.\venv\Scripts\Activate.ps1` (for PowerShell) or `.\venv\Scripts\activate.bat` (for CMD)
    *   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `torch` installation might require specific commands depending on your CUDA version if you plan to use GPU. Refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for instructions.*

4.  **Set Up Environment Variables**:
    *   Copy `.env.example` to a new file named `.env` in the project root:
        ```powershell
        Copy-Item .env.example .env # PowerShell
        ```
        or
        ```bash
        cp .env.example .env    # macOS/Linux/Git Bash
        ```
    *   Open the `.env` file and add your credentials:
        *   `OPENAI_API_KEY`: Your API key from OpenAI (if using `openai_api` mode).
        *   `HUGGING_FACE_ACCESS_TOKEN`: Your access token from Hugging Face (required for `pyannote.audio` models). You can create one [here](https://huggingface.co/settings/tokens).

---

## ⚙️ Configuration

The primary configuration file is `config.py`. You can modify it to change the behavior of the script:

*   `TRANSCRIPTION_MODE`: 
    *   `'local'`: Use Hugging Face Transformers for local transcription.
    *   `'openai_api'`: Use the OpenAI Whisper API.
*   `LOCAL_MODEL_ID`: Hugging Face model ID for local transcription (e.g., `"openai/whisper-large-v3"`).
*   `OPENAI_MODEL_NAME`: Model name for OpenAI API (e.g., `"whisper-1"`).
*   `DEVICE`: Computation device for local models (`"cuda"` for GPU, `"cpu"` for CPU).
*   `FFMPEG_EXECUTABLE_PATH`: Absolute path to your FFmpeg `bin` directory if it's not in the system PATH. 
    *   Example (Windows): `r"C:\path\to\ffmpeg\bin"`
    *   Example (Linux/macOS): `"/usr/local/bin/ffmpeg"` (often not needed if FFmpeg is installed via a package manager and in PATH).
*   `HUGGING_FACE_ACCESS_TOKEN`: Can also be set here as an alternative to the `.env` file, though `.env` is preferred for sensitive keys.

---

## ▶️ Usage

Run the main script from the root directory of the project:

```powershell
python main.py [audio_file_path] [output_csv_path]
```

**Arguments:**

*   `audio_file_path` (optional): Path to the input `.wav` audio file.
    *   Default: `harvard.wav/harvard.wav` (relative to the script location).
*   `output_csv_path` (optional): Path to save the output CSV file.
    *   Default: `output/output_YYYY-MM-DD_HH-MM-SS_AMPM.csv` (timestamped, in the `output` directory).

**Example:**

```powershell
python main.py "path\to\your\audio.wav" "path\to\your\output.csv"
```

If no arguments are provided, the script will use the default input audio (`harvard.wav/harvard.wav`) and generate a timestamped output CSV in the `output/` directory.

---

## 📊 Output Format

The script generates a CSV file with the following columns:

*   **Speaker**: Identifier for the speaker (e.g., `SPEAKER_00`, `SPEAKER_01`, or `Segment_X` if diarization fails or is not fully effective).
*   **Start Time**: Start time of the conversation segment in seconds.
*   **End Time**: End time of the conversation segment in seconds.
*   **Conversation**: The transcribed text for that segment.

**Example CSV Row:**

```csv
Speaker,Start Time,End Time,Conversation
SPEAKER_00,0.53,2.78,The stale smell of old beer lingers.
SPEAKER_01,3.01,5.52,It takes heat to bring out the odor.
```
*(Note: Timestamp accuracy for OpenAI API mode is currently under review and may show "N/A".)*

---

## 💡 Key Functionalities Explained

### Transcription

*   **Local Mode (`transformers`)**:
    *   Downloads and runs Whisper models directly on your machine.
    *   Offers more control over model versions and offline capability.
    *   Performance depends on your hardware (CPU/GPU).
    *   Uses `return_timestamps="segment"` for segment-level timestamps.
*   **OpenAI API Mode (`openai_api`)**:
    *   Sends audio to OpenAI's servers for transcription.
    *   Can be faster and may offer access to the latest model improvements without local setup.
    *   Requires an internet connection and an OpenAI API key.
    *   Requests `verbose_json` with `segment` and `word` granularities for timestamps. *(Current issue: Timestamps might appear as "N/A" in the output for this mode, which is being investigated.)*

### Speaker Diarization

*   Utilizes `pyannote.audio` library, specifically the `pyannote/speaker-diarization-pytorch` pretrained pipeline.
*   Processes the audio to identify different speakers and the time segments they speak.
*   The diarization results are then aligned with the transcription segments based on timestamp overlap to assign speaker labels.
*   Requires a Hugging Face access token for downloading the diarization model.

---

## 🗺️ Roadmap & Future Enhancements

*   ✅ **Fix Timestamp Extraction for OpenAI API Mode**: Ensure accurate segment start/end times are captured.
*   🌐 **Implement Translation Feature**: Add functionality to translate the transcribed text into specified languages.
*   🗣️ **Refine Speaker Diarization Alignment**: Improve the logic for matching diarization turns with transcription segments, especially for overlapping speech or short utterances.
*   🎧 **Support for More Audio Formats**: Extend compatibility beyond `.wav` files.
*   🖥️ **User Interface**: Develop a simple GUI (e.g., using Streamlit or Gradio) for easier interaction.
*   🧪 **Comprehensive Testing**: Thoroughly test with diverse audio samples, including mixed languages (English, Hokkien, Cantonese as per original requirements).
*   📄 **Detailed Logging**: Implement more robust logging for easier debugging and monitoring.

---

## ⭐ Show Your Support!

If you find WhisperCast useful, or just landed here while evading a particularly gnarly bug, consider giving this repo a 🌟 star! 

Why? 
- It helps other whisperers find this toolkit.
- It fuels the maintainers with virtual coffee (and occasional real coffee).
- It's scientifically proven* to increase your good code karma by at least 10 points!

(*Science pending peer review, but we're optimistic.)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes relevant tests if applicable.

---

## 📜 License

This project is currently unlicensed. Please specify a license (e.g., MIT, Apache 2.0) if you intend to distribute it broadly.

---

## 🙏 Acknowledgements

*   OpenAI for the Whisper model.
*   Hugging Face for the `transformers` library and model hosting.
*   The `pyannote.audio` team for their excellent speaker diarization toolkit.

