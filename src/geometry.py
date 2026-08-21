import math

from pose import LandmarkPoint


# Verifica se o landmark está visível antes de calcular o ângulo
def is_visible(landmark: LandmarkPoint, threshold: float) -> bool:
    if landmark.visibility is None:
        return True
    return landmark.visibility >= threshold

# Cálculo do ângulo entre dois landmarks
def joint_angle(
    proximal: LandmarkPoint | None,
    joint: LandmarkPoint | None,
    distal: LandmarkPoint | None,
    width: int,
    height: int,
) -> float | None:
    # Ângulo no plano da imagem: proximal → joint → distal (vetores em pixels).
    if proximal is None or joint is None or distal is None:
        return None

    v_proximal = (
        (proximal.x - joint.x) * width,
        (proximal.y - joint.y) * height,
    )
    v_distal = (
        (distal.x - joint.x) * width,
        (distal.y - joint.y) * height,
    )
    return _angle_between(v_proximal, v_distal)

# Cálculo do ângulo do cotovelo
def elbow_angle(
    shoulder: LandmarkPoint | None,
    elbow: LandmarkPoint | None,
    wrist: LandmarkPoint | None,
    width: int,
    height: int,
) -> float | None:
    return joint_angle(shoulder, elbow, wrist, width, height)

# Cálculo do ângulo do joelho
def knee_angle(
    hip: LandmarkPoint | None,
    knee: LandmarkPoint | None,
    ankle: LandmarkPoint | None,
    width: int,
    height: int,
) -> float | None:
    return joint_angle(hip, knee, ankle, width, height)

# Arredonda o ângulo para melhor visualização
def quantize_angle(value: float | None) -> int | float | None:
    if value is None:
        return None
    rounded = round(value, 1)
    if rounded == int(rounded):
        return int(rounded)
    return rounded

# Formata o ângulo para uma string
def format_angle(value: float | None) -> str | None:
    quantized = quantize_angle(value)
    if quantized is None:
        return None
    return str(quantized)

# Cálculo do ângulo entre dois vetores
def _angle_between(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float | None:
    norm_a = math.hypot(vector_a[0], vector_a[1])
    norm_b = math.hypot(vector_b[0], vector_b[1])
    if norm_a == 0 or norm_b == 0:
        return None
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))
