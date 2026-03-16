#!/usr/bin/env python3

from __future__ import annotations

import json
import sys

from app.batdetect2_backend import normalize_batdetect2_results


def main() -> int:
    request = json.load(sys.stdin)
    clip_path = request["clip_path"]

    try:
        from batdetect2 import api
    except ModuleNotFoundError:
        print(
            "batdetect2 is not installed in this environment. "
            "Use a Python 3.10 virtualenv or the Docker image.",
            file=sys.stderr,
        )
        return 2

    results = api.process_file(clip_path)
    normalized = normalize_batdetect2_results(results)
    normalized["notes"] = [
        "BatDetect2 is trained for UK bat species.",
        "Validate predictions against local known data before ecological use.",
    ]
    json.dump(normalized, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
