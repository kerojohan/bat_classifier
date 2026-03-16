from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class BackendConfigurationError(RuntimeError):
    """Raised when no classifier backend is configured."""


class BackendExecutionError(RuntimeError):
    """Raised when the configured backend fails or emits invalid output."""


@dataclass(frozen=True)
class ClassifierRequest:
    clip_path: Path
    original_filename: str
    start_seconds: float
    end_seconds: float
    sample_rate_hz: int
    duration_seconds: float

    def to_payload(self) -> dict[str, object]:
        return {
            "clip_path": str(self.clip_path),
            "original_filename": self.original_filename,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
        }


def invoke_classifier(command: tuple[str, ...], request: ClassifierRequest) -> dict[str, object]:
    if not command:
        raise BackendConfigurationError(
            "BAT_CLASSIFIER_COMMAND is not configured. Set it to an executable that reads JSON from stdin and writes JSON to stdout."
        )

    try:
        result = subprocess.run(
            list(command),
            input=json.dumps(request.to_payload()),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise BackendExecutionError(f"Classifier executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown classifier error"
        raise BackendExecutionError(stderr) from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BackendExecutionError("Classifier output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BackendExecutionError("Classifier output must be a JSON object")
    return payload
