#!/usr/bin/env python3

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    duration = float(payload["duration_seconds"])
    confidence = 0.55 if duration < 0.02 else 0.78
    response = {
        "species": "Pipistrellus kuhlii",
        "confidence": confidence,
        "top_k": [
            {"species": "Pipistrellus kuhlii", "confidence": confidence},
            {"species": "Pipistrellus pipistrellus", "confidence": 0.14},
            {"species": "Hypsugo savii", "confidence": 0.08},
        ],
        "notes": [
            "Mock backend for integration testing.",
            "Replace BAT_CLASSIFIER_COMMAND with your real classifier entrypoint.",
        ],
    }
    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
