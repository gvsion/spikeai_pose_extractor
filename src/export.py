import csv
from pathlib import Path

from geometry import elbow_angle, knee_angle, quantize_angle
from pose import LandmarkPoint, find_landmark

COORD_DECIMALS = 4
CSV_DELIMITER = ";"

# Cabeçalhos dos arquivos CSV
LANDMARK_HEADER = ("frame", "landmark", "x", "y", "z", "visibility")
ANGLE_HEADER = (
    "frame",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
)

# Arredonda o valor para o número de casas decimais especificado
def _round(value: float | None, decimals: int) -> float | None:
    if value is None:
        return None
    return round(value, decimals)

# Converte o valor para uma célula do CSV
def _cell(value: int | float | None) -> int | float | str:
    return "" if value is None else value

# Converte os valores para uma string
def _format_cell(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return str(value).replace(".", ",")
    return str(value)

# Converte os valores para uma lista de dicts
def _format_rows(rows: list[dict], fieldnames: tuple[str, ...]) -> list[dict]:
    formatted: list[dict] = []
    for row in rows:
        formatted.append({key: _format_cell(row.get(key)) for key in fieldnames})
    return formatted

# Converte os landmarks para uma lista de dicts com os valores arredondados
def landmark_rows(frame_index: int, landmarks: list[LandmarkPoint] | None) -> list[dict]:
    if not landmarks:
        return []

    rows: list[dict] = []
    for landmark in landmarks:
        rows.append(
            {
                "frame": frame_index,
                "landmark": landmark.name,
                "x": _round(landmark.x, COORD_DECIMALS),
                "y": _round(landmark.y, COORD_DECIMALS),
                "z": _round(landmark.z, COORD_DECIMALS),
                "visibility": _round(landmark.visibility, COORD_DECIMALS),
            }
        )
    return rows


# Converte os ângulos para um dict
def angle_row(
    frame_index: int,
    landmarks: list[LandmarkPoint] | None,
    width: int,
    height: int,
) -> dict:
    if not landmarks:
        return {
            "frame": frame_index,
            "left_elbow_angle": "",
            "right_elbow_angle": "",
            "left_knee_angle": "",
            "right_knee_angle": "",
        }

    return {
        "frame": frame_index,
        "left_elbow_angle": _cell(
            quantize_angle(
                elbow_angle(
                    find_landmark(landmarks, "LEFT_SHOULDER"),
                    find_landmark(landmarks, "LEFT_ELBOW"),
                    find_landmark(landmarks, "LEFT_WRIST"),
                    width,
                    height,
                )
            )
        ),
        "right_elbow_angle": _cell(
            quantize_angle(
                elbow_angle(
                    find_landmark(landmarks, "RIGHT_SHOULDER"),
                    find_landmark(landmarks, "RIGHT_ELBOW"),
                    find_landmark(landmarks, "RIGHT_WRIST"),
                    width,
                    height,
                )
            )
        ),
        "left_knee_angle": _cell(
            quantize_angle(
                knee_angle(
                    find_landmark(landmarks, "LEFT_HIP"),
                    find_landmark(landmarks, "LEFT_KNEE"),
                    find_landmark(landmarks, "LEFT_ANKLE"),
                    width,
                    height,
                )
            )
        ),
        "right_knee_angle": _cell(
            quantize_angle(
                knee_angle(
                    find_landmark(landmarks, "RIGHT_HIP"),
                    find_landmark(landmarks, "RIGHT_KNEE"),
                    find_landmark(landmarks, "RIGHT_ANKLE"),
                    width,
                    height,
                )
            )
        ),
    }

# Escreve os landmarks em um arquivo CSV
def write_landmarks_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LANDMARK_HEADER,
            delimiter=CSV_DELIMITER,
        )
        writer.writeheader()
        writer.writerows(_format_rows(rows, LANDMARK_HEADER))


# Escreve os ângulos em um arquivo CSV
def write_angles_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ANGLE_HEADER,
            delimiter=CSV_DELIMITER,
        )
        writer.writeheader()
        writer.writerows(_format_rows(rows, ANGLE_HEADER))
