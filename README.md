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
├── pyproject.toml      # Defines package metadata, dependencies, and build system
├── README.md           # This file
├── .env.example        # Example for environment variables (OpenAI API Key, Hugging Face Token)
├── output/             # Default directory for CSV outputs (ignored by Git)
├── src/
│   └── whispercast/
│       ├── __init__.py     # Makes 'whispercast' a package
│       ├── cli.py          # Command-line interface logic
│       ├── config.py       # Configuration for transcription mode, models, etc.
│       ├── core/
│       │   ├── __init__.py # Makes 'core' a sub-package
│       │   ├── transcription.py # Transcription logic
│       │   ├── diarization.py   # Diarization logic
│       │   └── processing.py    # Logic for processing and combining results
│       └── sample_data/
│           └── harvard.wav # Sample audio file
├── tests/                # Directory for pytest tests
│   ├── __init__.py
│   ├── test_cli.py
│   └── core/
│       ├── __init__.py
│       ├── test_transcription.py
│       ├── test_diarization.py
│       └── test_processing.py
└── requirements.txt    # List of dependencies (primarily for reference, pyproject.toml is canonical)
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
    git clone https://github.com/your-username/WhisperCast # Replace with your repo URL
    cd WhisperCast
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   Windows (PowerShell/CMD): `.\\\\venv\\\\Scripts\\\\Activate.ps1` (for PowerShell) or `.\\\\venv\\\\Scripts\\\\activate.bat` (for CMD)
    *   macOS/Linux: `source venv/bin/activate`

3.  **Install the Package**:
    For regular use:
    ```bash
    pip install .
    ```
    For development (editable install with testing dependencies):
    ```bash
    pip install -e .[dev]
    ```
    *Dependencies are managed in `pyproject.toml`.*
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
        *   `HUGGING_FACE_ACCESS_TOKEN`: Your access token from Hugging Face (required for `pyannote.audio` models if `--enable-diarization` is used). You can create one [here](https://huggingface.co/settings/tokens). Ensure you have accepted the terms of service for the diarization models (e.g., `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) on the Hugging Face website.

---

## ⚙️ Configuration

The primary configuration file is `src/whispercast/config.py`. You can modify it to change the behavior of the script:

*   `TRANSCRIPTION_MODE`: 
    *   `'local'`: Use Hugging Face Transformers for local transcription.
    *   `'openai_api'`: Use the OpenAI Whisper API.
*   `LOCAL_MODEL_ID`: Hugging Face model ID for local transcription (e.g., `"openai/whisper-large-v3"`).
*   `OPENAI_MODEL_NAME`: Model name for OpenAI API (e.g., `"whisper-1"`).
*   `DEVICE`: Computation device for local models (`"cuda"` for GPU, `"cpu"` for CPU).
*   `FFMPEG_EXECUTABLE_PATH`: Absolute path to your FFmpeg `bin` directory if it's not in the system PATH. 
    *   Example (Windows): `r"C:\\\\path\\\\to\\\\ffmpeg\\\\bin"`
    *   Example (Linux/macOS): `"/usr/local/bin/ffmpeg"` (often not needed if FFmpeg is installed via a package manager and in PATH).
*   `HUGGING_FACE_ACCESS_TOKEN`: Can also be set here as an alternative to the `.env` file for the diarization feature, though `.env` is preferred for sensitive keys.

---

## ▶️ Usage

Once the package is installed, you can use the `whispercast` command-line tool:

```bash
whispercast [AUDIO_FILE] [OUTPUT_CSV] [--enable-diarization] [--version]
```

**Positional Arguments:**

*   `audio_file` (optional): Path to the input audio file (e.g., `.wav`). 
    *   Default: Uses the `harvard.wav` sample included in the package (`src/whispercast/sample_data/harvard.wav`).
*   `output_csv` (optional): Path to save the output CSV file. 
    *   Default: `output/output_YYYY-MM-DD_HH-MM-SS_AMPM.csv` (timestamped, in an `output` directory created in the current working directory where you run the command).

**Optional Arguments:**

*   `--enable-diarization`: If provided, attempts speaker diarization using `pyannote.audio`.
    *   Requires a Hugging Face User Access Token (see Setup). 
    *   Ensure you have accepted the EULA for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face.
*   `--version`: Displays the version of WhisperCast and exits.
*   `-h, --help`: Show a help message and exit.

**Examples:**

1.  Transcribe the sample audio with default settings (no diarization):
    ```bash
    whispercast
    ```

2.  Transcribe a specific audio file and save to a specific CSV path, enabling diarization:
    ```bash
    whispercast "path/to/your/audio.wav" "path/to/your/output.csv" --enable-diarization
    ```

3.  Check the installed version:
    ```bash
    whispercast --version
    ```

---

## 📊 Output Format

The script generates a CSV file with the following columns:

*   **Speaker**: Identifier for the speaker (e.g., `SPEAKER_00`, `SPEAKER_01`). If diarization is not enabled or fails, this might be a generic segment identifier (e.g., `Segment_0`).
*   **Start Time**: Start time of the conversation segment in seconds.
*   **End Time**: End time of the conversation segment in seconds.
*   **Conversation**: The transcribed text for that segment.

**Example CSV Row (with diarization):**

```csv
Speaker,Start Time,End Time,Conversation
SPEAKER_00,0.53,2.78,The stale smell of old beer lingers.
SPEAKER_01,3.01,5.52,It takes heat to bring out the odor.
```

**Timestamp Handling:**
*   **Local Mode & Diarization Enabled (any mode):** Timestamps are generally segment-level based on Whisper's output or aligned with diarization segments.
*   **OpenAI API Mode (Diarization Disabled):** Word-level timestamps are utilized to provide more granular start and end times for segments. The system will still segment the transcript, but the segment boundaries will be based on word timings.

---

## 💡 Key Functionalities Explained

### Transcription

*   **Local Mode (`transformers`)**:
    *   Downloads and runs Whisper models (e.g., `openai/whisper-large-v3`) directly on your machine.
    *   Offers more control over model versions and offline capability.
    *   Performance depends on your hardware (CPU/GPU).
    *   Uses `return_timestamps="segments"` for segment-level timestamps.
*   **OpenAI API Mode (`openai_api`)**:
    *   Sends audio to OpenAI's servers for transcription using models like `whisper-1`.
    *   Can be faster and may offer access to the latest model improvements without local setup.
    *   Requires an internet connection and an OpenAI API key.
    *   Requests `timestamp_granularities=["word", "segment"]`. Word-level timestamps are used for segmenting if diarization is disabled; otherwise, segment-level or diarization-aligned timestamps are used.

### Speaker Diarization

*   Optionally enabled via the `--enable-diarization` flag.
*   Utilizes `pyannote.audio` library, specifically models like `pyannote/speaker-diarization-3.1`.
*   Processes the audio to identify different speakers and the time segments they speak.
*   The diarization results are then aligned with the transcription segments based on timestamp overlap to assign speaker labels.
*   Requires a Hugging Face access token and acceptance of model EULAs on Hugging Face.

---

## 🗺️ Roadmap & Future Enhancements

*   ✅ **Timestamp Accuracy & Granularity**: Continuously improve timestamp accuracy. Word-level timestamps from OpenAI API are now parsed when diarization is off. Further validation and refinement needed, especially in alignment with diarization.
*   🌐 **Implement Translation Feature**: Add functionality to translate the transcribed text into specified languages.
*   🗣️ **Refine Speaker Diarization Alignment**: Improve the logic for matching diarization turns with transcription segments, especially for overlapping speech or short utterances.
*   🎧 **Support for More Audio Formats**: Extend compatibility beyond `.wav` files (requires checking FFmpeg capabilities and potential library adjustments).
*   🖥️ **User Interface**: Develop a simple GUI (e.g., using Streamlit or Gradio) for easier interaction.
*   🧪 **Comprehensive Testing**: Thoroughly test with diverse audio samples, including mixed languages and noisy environments. Expand `pytest` suite.
*   📄 **Detailed Logging**: Implement more robust logging for easier debugging and monitoring.
*   📦 **Publish to PyPI**: Make the package easily installable via `pip install whispercast`.
*   Formalize versioning (e.g., using `git tags` and a tool like `setuptools_scm`).

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

