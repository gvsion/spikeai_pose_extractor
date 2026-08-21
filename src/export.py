import json
from pathlib import Path

from geometry import elbow_angle, quantize_angle
from pose import LandmarkPoint, find_landmark

COORD_DECIMALS = 4

# Arredonda o valor para o número de casas decimais especificado 
def _round(value: float | None, decimals: int) -> float | None:
    if value is None:
        return None
    return round(value, decimals)


# Função para registrar os landmarks e o ângulo do cotovelo
def frame_record(
    frame_index: int,
    landmarks: list[LandmarkPoint] | None,
    width: int,
    height: int,
) -> dict:
    if not landmarks:
        return {"frame": frame_index, "landmarks": [], "elbow_angle": None}

    return {
        "frame": frame_index,
        "landmarks": [
            {
                "name": landmark.name,
                "x": _round(landmark.x, COORD_DECIMALS),
                "y": _round(landmark.y, COORD_DECIMALS),
                "z": _round(landmark.z, COORD_DECIMALS),
                "visibility": _round(landmark.visibility, COORD_DECIMALS),
            }
            for landmark in landmarks
        ],
        "elbow_angle": {
            "left": quantize_angle(
                elbow_angle(
                    find_landmark(landmarks, "LEFT_SHOULDER"),
                    find_landmark(landmarks, "LEFT_ELBOW"),
                    find_landmark(landmarks, "LEFT_WRIST"),
                    width,
                    height,
                )
            ),
            "right": quantize_angle(
                elbow_angle(
                    find_landmark(landmarks, "RIGHT_SHOULDER"),
                    find_landmark(landmarks, "RIGHT_ELBOW"),
                    find_landmark(landmarks, "RIGHT_WRIST"),
                    width,
                    height,
                )
            ),
        },
    }

# Função para escrever os landmarks em um arquivo JSON
def write_landmarks_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
