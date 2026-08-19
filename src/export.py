"""Exporta os landmarks (dados) para JSON, um objeto por frame."""

import json
from pathlib import Path

from geometry import elbow_angle
from pose import LandmarkPoint, find_landmark


def frame_record(frame_index: int, landmarks: list[LandmarkPoint] | None) -> dict:
    if not landmarks:
        return {"frame": frame_index, "landmarks": [], "elbow_angle": None}

    return {
        "frame": frame_index,
        "landmarks": [
            {
                "name": landmark.name,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
            for landmark in landmarks
        ],
        "elbow_angle": {
            "left": elbow_angle(
                find_landmark(landmarks, "LEFT_SHOULDER"),
                find_landmark(landmarks, "LEFT_ELBOW"),
                find_landmark(landmarks, "LEFT_WRIST"),
            ),
            "right": elbow_angle(
                find_landmark(landmarks, "RIGHT_SHOULDER"),
                find_landmark(landmarks, "RIGHT_ELBOW"),
                find_landmark(landmarks, "RIGHT_WRIST"),
            ),
        },
    }


def write_landmarks_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
