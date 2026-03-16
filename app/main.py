from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import Settings
from app.service import (
    AudioProcessingError,
    BackendConfigurationError,
    BackendExecutionError,
    classify_audio_bytes,
)

app = FastAPI(title="Bat Classifier Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify")
async def classify(
    audio: UploadFile = File(...),
    start_seconds: float | None = Form(None),
    end_seconds: float | None = Form(None),
) -> dict[str, object]:
    settings = Settings.from_env()
    audio_bytes = await audio.read()

    try:
        return classify_audio_bytes(
            audio_bytes=audio_bytes,
            original_filename=audio.filename or "upload.bin",
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            settings=settings,
        )
    except AudioProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BackendConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BackendExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
