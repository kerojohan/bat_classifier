from __future__ import annotations

import os
from typing import Any

from app.audio import AudioProcessingError, cleanup_prepared_clip, prepare_clip
from app.backend import (
    BackendConfigurationError,
    BackendExecutionError,
    ClassifierRequest,
    invoke_classifier,
)
from app.config import Settings


def classify_audio_bytes(
    *,
    audio_bytes: bytes,
    original_filename: str,
    start_seconds: float | None,
    end_seconds: float | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    os.makedirs(settings.work_dir, exist_ok=True)

    prepared = prepare_clip(
        audio_bytes=audio_bytes,
        original_filename=original_filename,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        work_dir=settings.work_dir,
    )
    try:
        classifier_output = invoke_classifier(
            settings.classifier_command,
            ClassifierRequest(
                clip_path=prepared.clip_path,
                original_filename=prepared.audio.original_filename,
                start_seconds=prepared.selected_start_seconds,
                end_seconds=prepared.selected_end_seconds,
                sample_rate_hz=prepared.clip.sample_rate_hz,
                duration_seconds=prepared.clip.duration_seconds,
            ),
        )
        response: dict[str, Any] = {
            "request": {
                "filename": prepared.audio.original_filename,
                "start_seconds": prepared.selected_start_seconds,
                "end_seconds": prepared.selected_end_seconds,
            },
            "audio": {
                "duration_seconds": prepared.audio.duration_seconds,
                "sample_rate_hz": prepared.audio.sample_rate_hz,
                "channels": prepared.audio.channels,
                "codec": prepared.audio.codec,
            },
            "detection": {
                "mode": prepared.detection_mode,
                "candidates": [
                    {
                        "start_seconds": event.start_seconds,
                        "end_seconds": event.end_seconds,
                        "score": event.score,
                    }
                    for event in prepared.detected_events
                ],
            },
            "clip": {
                "duration_seconds": prepared.clip.duration_seconds,
                "sample_rate_hz": prepared.clip.sample_rate_hz,
                "channels": prepared.clip.channels,
                "frame_count": prepared.clip.frame_count,
                "peak_amplitude": prepared.clip.peak_amplitude,
                "rms_amplitude": prepared.clip.rms_amplitude,
            },
            "classifier": classifier_output,
        }
        if settings.keep_artifacts:
            response["clip"]["path"] = str(prepared.clip_path)
            response["artifacts"] = {"temp_dir": str(prepared.temp_dir)}
        return response
    finally:
        if not settings.keep_artifacts:
            cleanup_prepared_clip(prepared)


__all__ = [
    "AudioProcessingError",
    "BackendConfigurationError",
    "BackendExecutionError",
    "classify_audio_bytes",
]
