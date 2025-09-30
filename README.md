# Parakeet Whisper-Compatible API

A simple FastAPI server that provides an OpenAI Whisper API-compatible endpoint backed by [NVIDIA's Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for speech recognition + [Pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization.

## Features

- Complete drop-in replacement for OpenAI's Whisper API
- Uses [NVIDIA's Parakeet-TDT 0.6B V3 model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for high-quality multilingual transcription
- Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
- Supports word-level and segment-level timestamps
- Optional speaker diarization using [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- FastAPI-based server with automatic OpenAPI documentation

## About Parakeet-TDT 0.6B V3

This project uses [NVIDIA's Parakeet-TDT-0.6B-V3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a state-of-the-art multilingual automatic speech recognition (ASR) model:

### Key Specifications
- **Model Size**: 600 million parameters
- **Architecture**: FastConformer-TDT
- **Languages**: 25 European languages with automatic detection
  - Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian
- **Training Data**: 660,000+ hours from Granary multilingual corpus + 10,000 hours human-transcribed data
- **License**: CC-BY-4.0 (commercial and non-commercial use allowed)

### Performance Highlights
- **Word Error Rate**: 1.93% (LibriSpeech test-clean) to 6.34% average across 8 benchmarks
- **Features**: Automatic punctuation, capitalization, word-level and segment-level timestamps
- **Long Audio Support**: Up to 24 minutes with full attention, 3+ hours with local attention
- **Hardware**: Optimized for NVIDIA GPUs (Volta, Ampere, Hopper, Blackwell architectures)

### Multilingual Benchmarks (WER %)

| Language | Fleurs | MLS | CoVoST |
|----------|--------|-----|--------|
| English  | 4.85%  | -   | 6.80%  |
| German   | 5.04%  | -   | 4.84%  |
| Spanish  | 3.45%  | 4.39% | 3.41% |
| French   | 5.15%  | 4.97% | 6.05% |
| Italian  | 3.00%  | 10.08% | 3.69% |
| Polish   | 7.31%  | 7.28% | -     |
| Russian  | 5.51%  | -   | 3.00%  |

**Average WER**: 11.97% (Fleurs), 7.83% (MLS), 11.98% (CoVoST)

*Full model details and technical report: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3*

## Requirements

- NVIDIA GPU with CUDA support (recommended)
- Python 3.8 or higher (Python 3.10-3.11 recommended for best compatibility)
- ffmpeg (for audio processing)
- HuggingFace account and access token (required for speaker diarization)

**Note**: If using conda/miniconda, ensure you have updated C++ standard library (`libstdc++`). For Python 3.13+, system Python is recommended to avoid library conflicts.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jfgonsalves/parakeet-diarized
   cd parakeet-diarized
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up speaker diarization (optional):
   - Create a free account at [HuggingFace](https://huggingface.co/)
   - Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Accept the user agreement for the [Pyannote speaker diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1)

5. Run the server:

   **With speaker diarization:**
   ```bash
   ./run.sh --hf-token "your_token_here"
   ```

   **Without speaker diarization:**
   ```bash
   ./run.sh
   ```

   **Other options:**
   ```bash
   ./run.sh --help  # See all available options
   ./run.sh --port 8080 --debug --hf-token "your_token_here"
   ```

## Usage

### API Endpoints

The API mimics the OpenAI Whisper API interface:

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Parameters:
- `file`: The audio file to transcribe (multipart/form-data)
- `model`: Model to use (defaults to "whisper-1", but will use Parakeet regardless)
- `language`: Language of the audio (optional)
- `response_format`: Format of the response (defaults to "json", options: json, text, srt, vtt, verbose_json)
- `timestamps`: Whether to include timestamps (defaults to false)
- `timestamp_granularities`: Timestamp detail level (accepts "segment")
- `temperature`: Temperature for sampling (defaults to 0.0)
- `vad_filter`: Voice activity detection filter (defaults to false)
- `prompt`: Optional prompt to guide the transcription (ignored but accepted for compatibility)
- `diarize`: Enable speaker diarization (defaults to true, requires HuggingFace token)
- `include_diarization_in_text`: Include speaker labels in transcript text (defaults to true)

Example with curl:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

#### Health Check

```
GET /health
```

Returns the health status of the API and the loaded model.

## Compatibility with OpenAI Whisper API

This API is designed to be a drop-in replacement for the OpenAI Whisper API:

1. Supports all Whisper API response formats (json, text, srt, vtt, verbose_json)
2. Accepts all major Whisper API parameters for compatibility
3. Returns responses in the same format as the OpenAI Whisper API
4. Provides a `/v1/models` endpoint for application compatibility

Minor differences:
1. The `model` parameter is accepted but ignored - always uses Parakeet-TDT
2. Some advanced Whisper-specific parameters might have no effect
3. Performance characteristics may differ from OpenAI's implementation

## API Response Formats

The API supports multiple response formats:

### JSON (default)
```json
{
  "text": "Full transcription text goes here"
}
```

### Verbose JSON
```json
{
  "text": "Full transcription text goes here",
  "task": "transcribe",
  "language": "en",
  "duration": 10.5,
  "model": "parakeet-tdt-0.6b-v3",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Segment text",
      "tokens": [50364, 2425, 286, 257],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.1
    },
    {
      "id": 1,
      "start": 2.5,
      "end": 5.0,
      "text": "Another segment",
      "tokens": [50364, 5816, 2121],
      "temperature": 0.0,
      "avg_logprob": -0.6,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.05
    }
  ]
}
```

### Plain Text
```
Full transcription text goes here
```

### SRT
```
1
00:00:00,000 --> 00:00:02,500
Segment text

2
00:00:02,500 --> 00:00:05,000
Another segment
```

### VTT
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Segment text

00:00:02.500 --> 00:00:05.000
Another segment
```

The `segments` field is included when the `timestamps` parameter is set to `true` or when using `verbose_json` format.

## Speaker Diarization

The API includes speaker diarization capabilities using [Pyannote.audio](https://github.com/pyannote/pyannote-audio):

### Setup Requirements

For speaker diarization to work, you need:

1. **HuggingFace Account**: Create a free account at [huggingface.co](https://huggingface.co/)
2. **Access Token**: Generate a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
3. **Model Agreement**: Accept the user agreement for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **Environment Variable**: Set `HUGGINGFACE_ACCESS_TOKEN` with your token

### Features

- Automatic speaker detection and labeling
- Integration with transcription segments
- Optional speaker labels in transcript text
- Support for multiple speakers per audio file

### Usage

Enable diarization by setting `diarize=true` in your API request:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F diarize=true \
  -F include_diarization_in_text=true
```

When `include_diarization_in_text=true`, the transcript will include speaker labels:
```
Speaker 1: Hello, how are you today?
Speaker 2: I'm doing well, thank you for asking.
```

### Configuration

Use the `run.sh` script to configure and start the server:

```bash
./run.sh --help
# Options:
#   --debug             Enable debug mode
#   --port PORT         Set server port (default: 8000)
#   --host HOST         Set server host (default: 0.0.0.0)
#   --skip-deps-check   Skip dependency checking
#   --hf-token TOKEN    Set HuggingFace access token for speaker diarization
#   --help              Show help message
```

**Environment Variables** (for settings not available as command line arguments):
- `ENABLE_DIARIZATION`: Enable/disable diarization globally (default: true)
- `INCLUDE_DIARIZATION_IN_TEXT`: Include speaker labels in text by default (default: true)
- `MODEL_ID`: Parakeet model to use (default: nvidia/parakeet-tdt-0.6b-v3)
- `TEMPERATURE`: Sampling temperature (default: 0.0)
- `CHUNK_DURATION`: Audio chunk duration in seconds (default: 500)
- `TEMP_DIR`: Temporary directory for audio processing (default: /tmp/parakeet)

## Performance

### ASR Transcription
The Parakeet-TDT-0.6B-V3 model delivers:
- **Speed**: Fastest multilingual model on HuggingFace Open ASR Leaderboard
- **Throughput**: Transcribes 1 hour of audio in ~seconds (GPU-dependent)
- **Accuracy**: Achieves state-of-the-art WER across 25 European languages (see model details above)
- **Quality**: Automatic punctuation, capitalization, and accurate timestamps included

### Speaker Diarization
[Pyannote.audio](https://github.com/pyannote/pyannote-audio) adds:
- Automatic speaker identification using state-of-the-art models
- Real-time speaker change detection
- Support for unlimited number of speakers
- Integration with transcription segments for seamless speaker-labeled output

## Acknowledgments

This project builds upon excellent work by:

- **NVIDIA NeMo Team**: For the outstanding [Parakeet-TDT model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) that provides state-of-the-art multilingual speech recognition
- **Pyannote Team**: For the powerful [Pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization toolkit

## License

This project is released under MIT License. However, the Parakeet-TDT model is governed by the CC-BY-4.0 license.
