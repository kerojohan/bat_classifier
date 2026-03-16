from __future__ import annotations

from collections import defaultdict
from typing import Any


def normalize_batdetect2_results(results: dict[str, Any]) -> dict[str, Any]:
    pred_dict = results.get("pred_dict") or {}
    annotations = pred_dict.get("annotation") or []

    species_scores: dict[str, float] = defaultdict(float)
    top_detection_per_species: dict[str, dict[str, Any]] = {}
    normalized_detections: list[dict[str, Any]] = []

    for raw_detection in annotations:
        species = str(raw_detection.get("class") or "Unknown")
        class_prob = float(raw_detection.get("class_prob") or 0.0)
        det_prob = float(raw_detection.get("det_prob") or 0.0)
        combined_score = class_prob * det_prob
        existing_score = species_scores.get(species, 0.0)
        if combined_score > existing_score:
            species_scores[species] = combined_score
            top_detection_per_species[species] = {
                "start_time": float(raw_detection.get("start_time") or 0.0),
                "end_time": float(raw_detection.get("end_time") or 0.0),
                "low_freq": raw_detection.get("low_freq"),
                "high_freq": raw_detection.get("high_freq"),
                "class_prob": class_prob,
                "det_prob": det_prob,
                "event": raw_detection.get("event"),
            }

        normalized_detections.append(
            {
                "start_time": float(raw_detection.get("start_time") or 0.0),
                "end_time": float(raw_detection.get("end_time") or 0.0),
                "low_freq": raw_detection.get("low_freq"),
                "high_freq": raw_detection.get("high_freq"),
                "species": species,
                "class_prob": class_prob,
                "det_prob": det_prob,
                "combined_confidence": combined_score,
                "event": raw_detection.get("event"),
            }
        )

    sorted_species = sorted(species_scores.items(), key=lambda item: item[1], reverse=True)
    top_species = sorted_species[0][0] if sorted_species else "No bat detected"
    top_confidence = sorted_species[0][1] if sorted_species else 0.0

    return {
        "species": top_species,
        "confidence": top_confidence,
        "top_k": [
            {
                "species": species,
                "confidence": confidence,
                "best_detection": top_detection_per_species[species],
            }
            for species, confidence in sorted_species[:5]
        ],
        "detections": normalized_detections,
        "detection_count": len(normalized_detections),
        "model": {
            "name": "BatDetect2",
            "scope": "UK bat species",
            "license": "CC BY-NC 4.0",
        },
        "raw": {
            "id": pred_dict.get("id"),
        },
    }
