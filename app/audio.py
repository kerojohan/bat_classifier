from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioProcessingError(RuntimeError):
    """Raised when the uploaded file cannot be inspected or clipped."""


@dataclass(frozen=True)
class AudioMetadata:
    original_filename: str
    duration_seconds: float
    sample_rate_hz: int | None
    channels: int | None
    codec: str | None


@dataclass(frozen=True)
class ClipFeatures:
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    frame_count: int
    peak_amplitude: float
    rms_amplitude: float


@dataclass(frozen=True)
class PreparedClip:
    temp_dir: Path
    input_path: Path
    clip_path: Path
    selected_start_seconds: float
    selected_end_seconds: float
    detection_mode: str
    detected_events: tuple["DetectedEvent", ...]
    audio: AudioMetadata
    clip: ClipFeatures


@dataclass(frozen=True)
class DetectedEvent:
    start_seconds: float
    end_seconds: float
    score: float


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioProcessingError(f"Missing required executable: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        raise AudioProcessingError(stderr) from exc


def _probe_audio(input_path: Path, original_filename: str) -> AudioMetadata:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise AudioProcessingError("Audio duration could not be determined")
    sample_rate = stream.get("sample_rate")
    return AudioMetadata(
        original_filename=original_filename,
        duration_seconds=duration,
        sample_rate_hz=int(sample_rate) if sample_rate else None,
        channels=stream.get("channels"),
        codec=stream.get("codec_name"),
    )


def prepare_clip(
    *,
    audio_bytes: bytes,
    original_filename: str,
    start_seconds: float | None,
    end_seconds: float | None,
    work_dir: str,
) -> PreparedClip:
    suffix = Path(original_filename or "upload.bin").suffix or ".bin"
    temp_dir = Path(tempfile.mkdtemp(prefix="bat-", dir=work_dir))
    input_path = temp_dir / f"input{suffix}"
    input_path.write_bytes(audio_bytes)

    audio_meta = _probe_audio(input_path, original_filename)
    detected_events: tuple[DetectedEvent, ...] = ()
    detection_mode = "manual"

    if start_seconds is None and end_seconds is None:
        detected_events = detect_events(input_path, audio_meta.duration_seconds)
        if not detected_events:
            raise AudioProcessingError("No candidate bat events detected in the audio")
        selected = detected_events[0]
        start_seconds = selected.start_seconds
        end_seconds = selected.end_seconds
        detection_mode = "auto"
    elif start_seconds is None or end_seconds is None:
        raise AudioProcessingError("start_seconds and end_seconds must both be provided, or both omitted for auto-detection")
    else:
        if start_seconds < 0:
            raise AudioProcessingError("start_seconds must be >= 0")
        if end_seconds <= start_seconds:
            raise AudioProcessingError("end_seconds must be greater than start_seconds")
        if end_seconds > audio_meta.duration_seconds:
            raise AudioProcessingError(
                f"Requested interval [{start_seconds}, {end_seconds}] exceeds audio duration {audio_meta.duration_seconds:.3f}s"
            )

    clip_path = temp_dir / "clip.wav"
    _run_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.6f}",
            "-to",
            f"{end_seconds:.6f}",
            "-i",
            str(input_path),
            "-acodec",
            "pcm_s16le",
            str(clip_path),
        ]
    )

    data, sample_rate = sf.read(str(clip_path))
    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim == 1:
        channels = 1
        mono = samples
    else:
        channels = int(samples.shape[1])
        mono = samples.mean(axis=1)

    frame_count = int(mono.shape[0])
    if frame_count == 0:
        raise AudioProcessingError("The requested interval produced an empty clip")

    clip = ClipFeatures(
        duration_seconds=frame_count / float(sample_rate),
        sample_rate_hz=int(sample_rate),
        channels=channels,
        frame_count=frame_count,
        peak_amplitude=float(np.max(np.abs(mono))),
        rms_amplitude=float(np.sqrt(np.mean(np.square(mono)))),
    )
    return PreparedClip(
        temp_dir=temp_dir,
        input_path=input_path,
        clip_path=clip_path,
        selected_start_seconds=start_seconds,
        selected_end_seconds=end_seconds,
        detection_mode=detection_mode,
        detected_events=detected_events,
        audio=audio_meta,
        clip=clip,
    )


def cleanup_prepared_clip(prepared: PreparedClip) -> None:
    shutil.rmtree(prepared.temp_dir, ignore_errors=True)


def detect_events(
    input_path: Path,
    duration_seconds: float,
    *,
    analysis_window_ms: float = 2.0,
    min_event_ms: float = 6.0,
    max_gap_ms: float = 8.0,
    padding_ms: float = 4.0,
    max_events: int = 5,
) -> tuple[DetectedEvent, ...]:
    try:
        with sf.SoundFile(str(input_path)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            if sample_rate <= 0:
                raise AudioProcessingError("Audio sample rate could not be determined for auto-detection")

            window_frames = max(1, int(sample_rate * analysis_window_ms / 1000.0))
            block_frames = window_frames * 2048
            rms_chunks: list[np.ndarray] = []
            remainder = np.empty(0, dtype=np.float32)

            while True:
                block = audio_file.read(block_frames, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                mono = block.mean(axis=1)
                if remainder.size:
                    mono = np.concatenate((remainder, mono))
                full_window_count = mono.shape[0] // window_frames
                if full_window_count:
                    windowed = mono[: full_window_count * window_frames].reshape(full_window_count, window_frames)
                    windowed64 = windowed.astype(np.float64, copy=False)
                    rms_chunks.append(np.sqrt(np.mean(np.square(windowed64), axis=1)))
                remainder = mono[full_window_count * window_frames :]

            if remainder.size:
                padded = np.zeros(window_frames, dtype=np.float64)
                padded[: remainder.size] = remainder
                rms_chunks.append(np.array([float(np.sqrt(np.mean(np.square(padded))))], dtype=np.float64))
    except RuntimeError as exc:
        raise AudioProcessingError("Audio format is not supported for auto-detection") from exc

    if not rms_chunks:
        return ()

    energies = np.concatenate(rms_chunks)
    if not np.any(energies > 0):
        return ()

    p50 = float(np.percentile(energies, 50))
    p75 = float(np.percentile(energies, 75))
    p99 = float(np.percentile(energies, 99))
    threshold = max(p75, p50 + (p99 - p50) * 0.30)
    mask = energies >= threshold

    max_gap_windows = max(1, int(round(max_gap_ms / analysis_window_ms)))
    min_event_windows = max(1, int(round(min_event_ms / analysis_window_ms)))
    padding_windows = max(1, int(round(padding_ms / analysis_window_ms)))

    active_indices = np.flatnonzero(mask)
    segments: list[tuple[int, int]] = []
    if active_indices.size:
        start_index = int(active_indices[0])
        previous = start_index
        for index in active_indices[1:]:
            current = int(index)
            if current - previous <= max_gap_windows:
                previous = current
                continue
            segments.append((start_index, previous))
            start_index = current
            previous = current
        segments.append((start_index, previous))

    events: list[DetectedEvent] = []
    for start_index, end_index in segments:
        if end_index - start_index + 1 < min_event_windows:
            continue
        padded_start = max(0, start_index - padding_windows)
        padded_end = min(len(energies) - 1, end_index + padding_windows)
        start_seconds = max(0.0, padded_start * analysis_window_ms / 1000.0)
        end_seconds = min(duration_seconds, (padded_end + 1) * analysis_window_ms / 1000.0)
        score = float(np.max(energies[start_index : end_index + 1]))
        events.append(
            DetectedEvent(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                score=score,
            )
        )

    if not events:
        peak_index = int(np.argmax(energies))
        start_seconds = max(0.0, (peak_index - padding_windows) * analysis_window_ms / 1000.0)
        end_seconds = min(duration_seconds, (peak_index + padding_windows + 1) * analysis_window_ms / 1000.0)
        return (
            DetectedEvent(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                score=float(energies[peak_index]),
            ),
        )

    events.sort(key=lambda event: event.score, reverse=True)
    return tuple(events[:max_events])
