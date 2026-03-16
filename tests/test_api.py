from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.batdetect2_backend import normalize_batdetect2_results
from app.cli import main as cli_main


def build_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 48000) -> bytes:
    frame_total = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for frame_index in range(frame_total):
            value = int(20000 * math.sin(2 * math.pi * 1000 * frame_index / sample_rate))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


def build_detectable_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 48000) -> bytes:
    frame_total = int(duration_seconds * sample_rate)
    event_start = int(0.40 * sample_rate)
    event_end = int(0.50 * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for frame_index in range(frame_total):
            amplitude = 1500
            if event_start <= frame_index < event_end:
                amplitude = 28000
            value = int(amplitude * math.sin(2 * math.pi * 1500 * frame_index / sample_rate))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


class ClassifyApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["BAT_WORK_DIR"] = self.temp_dir.name
        os.environ["BAT_CLASSIFIER_COMMAND"] = f"{sys.executable} tests/fixtures/mock_classifier.py"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        os.environ.pop("BAT_WORK_DIR", None)
        os.environ.pop("BAT_CLASSIFIER_COMMAND", None)

    def test_classify_returns_json_with_clip_and_classifier_data(self) -> None:
        response = self.client.post(
            "/classify",
            files={"audio": ("bat.wav", build_wav_bytes(), "audio/wav")},
            data={"start_seconds": "0.10", "end_seconds": "0.25"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classifier"]["species"], "Pipistrellus kuhlii")
        self.assertAlmostEqual(payload["clip"]["duration_seconds"], 0.15, places=2)
        self.assertNotIn("path", payload["clip"])
        self.assertEqual(payload["detection"]["mode"], "manual")

    def test_classify_keeps_artifacts_when_requested(self) -> None:
        os.environ["BAT_KEEP_ARTIFACTS"] = "true"
        response = self.client.post(
            "/classify",
            files={"audio": ("bat.wav", build_wav_bytes(), "audio/wav")},
            data={"start_seconds": "0.10", "end_seconds": "0.25"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(Path(payload["clip"]["path"]).exists())
        self.assertTrue(Path(payload["artifacts"]["temp_dir"]).exists())
        os.environ.pop("BAT_KEEP_ARTIFACTS", None)

    def test_classify_auto_detects_event_when_range_is_omitted(self) -> None:
        response = self.client.post(
            "/classify",
            files={"audio": ("bat.wav", build_detectable_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["detection"]["mode"], "auto")
        self.assertGreaterEqual(len(payload["detection"]["candidates"]), 1)
        self.assertGreater(payload["request"]["start_seconds"], 0.35)
        self.assertLess(payload["request"]["start_seconds"], 0.48)
        self.assertGreater(payload["request"]["end_seconds"], 0.45)
        self.assertLess(payload["request"]["end_seconds"], 0.58)

    def test_classify_rejects_invalid_ranges(self) -> None:
        response = self.client.post(
            "/classify",
            files={"audio": ("bat.wav", build_wav_bytes(), "audio/wav")},
            data={"start_seconds": "0.50", "end_seconds": "0.10"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("greater than start_seconds", response.json()["detail"])

    def test_classify_requires_backend_configuration(self) -> None:
        os.environ.pop("BAT_CLASSIFIER_COMMAND", None)
        response = self.client.post(
            "/classify",
            files={"audio": ("bat.wav", build_wav_bytes(), "audio/wav")},
            data={"start_seconds": "0.10", "end_seconds": "0.20"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("BAT_CLASSIFIER_COMMAND", response.json()["detail"])

    def test_cli_outputs_json(self) -> None:
        audio_path = Path(self.temp_dir.name) / "bat.wav"
        audio_path.write_bytes(build_wav_bytes())

        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            exit_code = cli_main([str(audio_path), "--start", "0.10", "--end", "0.20"])
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["classifier"]["species"], "Pipistrellus kuhlii")
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_auto_detects_event_without_explicit_range(self) -> None:
        audio_path = Path(self.temp_dir.name) / "bat_auto.wav"
        audio_path.write_bytes(build_detectable_wav_bytes())

        stdout = io.StringIO()
        stderr = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            exit_code = cli_main([str(audio_path)])
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["detection"]["mode"], "auto")
        self.assertGreaterEqual(len(payload["detection"]["candidates"]), 1)

    def test_batdetect2_result_normalization(self) -> None:
        payload = normalize_batdetect2_results(
            {
                "pred_dict": {
                    "id": "clip.wav",
                    "annotation": [
                        {
                            "start_time": 0.1,
                            "end_time": 0.12,
                            "low_freq": 20000,
                            "high_freq": 45000,
                            "class": "Pipistrellus pipistrellus",
                            "class_prob": 0.8,
                            "det_prob": 0.9,
                            "event": "Echolocation",
                        },
                        {
                            "start_time": 0.2,
                            "end_time": 0.25,
                            "low_freq": 18000,
                            "high_freq": 42000,
                            "class": "Myotis myotis",
                            "class_prob": 0.6,
                            "det_prob": 0.95,
                            "event": "Echolocation",
                        },
                    ],
                }
            }
        )

        self.assertEqual(payload["species"], "Pipistrellus pipistrellus")
        self.assertEqual(payload["detection_count"], 2)
        self.assertEqual(payload["top_k"][0]["species"], "Pipistrellus pipistrellus")
        self.assertEqual(payload["model"]["name"], "BatDetect2")


if __name__ == "__main__":
    unittest.main()
