import math

from pose import LandmarkPoint


def is_visible(landmark: LandmarkPoint, threshold: float) -> bool:
    if landmark.visibility is None:
        return True
    return landmark.visibility >= threshold


def elbow_angle(
    shoulder: LandmarkPoint | None,
    elbow: LandmarkPoint | None,
    wrist: LandmarkPoint | None,
) -> float | None:
    if shoulder is None or elbow is None or wrist is None:
        return None

    v_shoulder = (shoulder.x - elbow.x, shoulder.y - elbow.y)
    v_wrist = (wrist.x - elbow.x, wrist.y - elbow.y)
    return _angle_between(v_shoulder, v_wrist)


def quantize_angle(value: float | None) -> int | float | None:
    if value is None:
        return None
    rounded = round(value, 1)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def format_angle(value: float | None) -> str | None:
    quantized = quantize_angle(value)
    if quantized is None:
        return None
    return str(quantized)


def _angle_between(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float | None:
    norm_a = math.hypot(vector_a[0], vector_a[1])
    norm_b = math.hypot(vector_b[0], vector_b[1])
    if norm_a == 0 or norm_b == 0:
        return None
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))
