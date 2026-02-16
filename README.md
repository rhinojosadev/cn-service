# Chinese Transcription API

FastAPI service that accepts uploaded audio, transcribes spoken Chinese with OpenAI Whisper, and returns both Hanzi and pinyin.

## Features

- `POST /transcribe` endpoint for multipart audio uploads
- Optional WAV silence trimming before transcription
- Pinyin conversion from transcribed Chinese text
- API key support via form field (`api_key`) or environment variable (`OPENAI_API_KEY`)
- Health check endpoint at `GET /health`

## Tech Stack

- FastAPI
- Uvicorn
- OpenAI Python SDK
- NumPy
- pypinyin

## Requirements

- Python 3.10+ recommended
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Service

From the project root:

```bash
fastapi dev src/main.py
```

PowerShell with local virtual environment:

```powershell
.\.venv\Scripts\python -m fastapi dev src/main.py
```

The API will be available at:

- `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

## Configuration

Set your API key with either option:

1. Environment variable:

```bash
export OPENAI_API_KEY="your_key_here"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

2. Request form field:

- `api_key`: your OpenAI API key

## API Endpoints

### `GET /health`

Returns:

```json
{"status":"ok"}
```

### `POST /transcribe`

Accepts `multipart/form-data`:

- `audio` (required): audio file upload
- `language` (optional): defaults to `zh`
- `temperature` (optional): defaults to `0.0`
- `api_key` (optional): OpenAI API key override

Example request:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@sample.wav" \
  -F "language=zh" \
  -F "temperature=0.0"
```

Example response:

```json
{
  "text": "ni hao shi jie",
  "pinyin": "ni3 hao3 shi4 jie4",
  "filename": "sample.wav",
  "content_type": "audio/wav",
  "trimmed": true
}
```

## Notes

- Silence trimming runs only when the uploaded file is WAV (`RIFF/WAVE`).
- If no API key is provided in the request or environment, the API returns HTTP `400`.
- Unexpected transcription failures return HTTP `500` with an error message.
