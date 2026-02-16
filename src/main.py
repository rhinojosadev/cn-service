from io import BytesIO
import os
import struct
import traceback
import wave

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI
from pypinyin import Style, pinyin

app = FastAPI(title="Chinese Transcription API")


def trim_silence(raw_bytes: bytes, threshold: float = 0.02) -> bytes:
    """Remove leading/trailing silence from WAV audio bytes."""
    with wave.open(BytesIO(raw_bytes), "rb") as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(wav_file.getnframes())

    sample_width = params.sampwidth
    if sample_width not in (1, 2, 4):
        return raw_bytes

    fmt = {1: "b", 2: "h", 4: "i"}[sample_width]
    samples = struct.unpack(f"<{len(frames) // sample_width}{fmt}", frames)
    arr = np.array(samples, dtype=np.float32)
    arr /= max(abs(arr.max()), abs(arr.min()), 1)

    mask = np.abs(arr) > threshold
    if not mask.any():
        return raw_bytes

    first = int(mask.argmax())
    last = int(len(mask) - mask[::-1].argmax())
    margin = params.framerate // 10  # 100ms margin
    first = max(0, first - margin)
    last = min(len(arr), last + margin)

    trimmed = struct.pack(f"<{last - first}{fmt}", *samples[first:last])
    out = BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(trimmed)
    return out.getvalue()


def is_wav_bytes(raw_bytes: bytes) -> bool:
    return raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE"


def get_openai_client(api_key_from_request: str | None) -> OpenAI:
    api_key = api_key_from_request or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing API key. Pass form field 'api_key' or set OPENAI_API_KEY.",
        )
    return OpenAI(api_key=api_key)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    temperature: float = Form(0.0),
    api_key: str | None = Form(default=None),
):
    """
    Accept audio from React Native as multipart/form-data and return Hanzi + pinyin.

    React Native example field name:
    - audio: file object
    """
    try:
        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Keep your existing silence trim behavior for WAV input.
        processed_bytes = trim_silence(raw_bytes) if is_wav_bytes(raw_bytes) else raw_bytes

        audio_file = BytesIO(processed_bytes)
        audio_file.name = audio.filename or "audio.wav"

        client = get_openai_client(api_key)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            temperature=temperature,
        )

        text = transcript.text.strip()
        py = " ".join(item[0] for item in pinyin(text, style=Style.TONE)) if text else ""

        return JSONResponse(
            {
                "text": text,
                "pinyin": py,
                "filename": audio.filename,
                "content_type": audio.content_type,
                "trimmed": is_wav_bytes(raw_bytes),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)