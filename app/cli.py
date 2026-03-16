from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import Settings
from app.service import (
    AudioProcessingError,
    BackendConfigurationError,
    BackendExecutionError,
    classify_audio_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a bat call segment from a full audio file and print JSON."
    )
    parser.add_argument("audio_path", help="Path to the source audio file")
    parser.add_argument("--start", dest="start_seconds", type=float, help="Start time in seconds")
    parser.add_argument("--end", dest="end_seconds", type=float, help="End time in seconds")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep extracted clip files and include artifact paths in the JSON output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audio_path = Path(args.audio_path)
    settings = Settings.from_env()
    if args.keep_artifacts:
        settings = Settings(
            classifier_command=settings.classifier_command,
            work_dir=settings.work_dir,
            keep_artifacts=True,
        )

    try:
        payload = classify_audio_bytes(
            audio_bytes=audio_path.read_bytes(),
            original_filename=audio_path.name,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            settings=settings,
        )
    except FileNotFoundError:
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1
    except (AudioProcessingError, BackendConfigurationError, BackendExecutionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
