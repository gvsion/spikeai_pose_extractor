import numpy as np
import cv2

from geometry import elbow_angle, format_angle, is_visible, knee_angle
from pose import LandmarkPoint, find_landmark

DEFAULT_VISIBILITY_THRESHOLD = 0.5

# Conexões entre landmarks do esqueleto
SKELETON_CONNECTIONS = (
    ("HEAD", "LEFT_SHOULDER"),
    ("HEAD", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
)

# Renderização da pose
class PoseRenderer:
    def __init__(
        self,
        point_radius: int = 4,
        line_thickness: int = 2,
        visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    ) -> None:
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.visibility_threshold = visibility_threshold
        self._connections = SKELETON_CONNECTIONS

    def empty_frame(self, width: int, height: int) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def render(
        self,
        landmarks: list[LandmarkPoint] | None,
        width: int,
        height: int,
        base_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        canvas = base_frame.copy() if base_frame is not None else self.empty_frame(width, height)
        if not landmarks:
            return canvas

        by_name = {landmark.name: landmark for landmark in landmarks}
        self._draw_connections(canvas, by_name, width, height)
        self._draw_points(canvas, landmarks, width, height)
        self._draw_joint_angles(canvas, landmarks, width, height)
        return canvas

    # Desenha os ângulos dos cotovelos e joelhos
    def _draw_joint_angles(
        self,
        canvas: np.ndarray,
        landmarks: list[LandmarkPoint],
        width: int,
        height: int,
    ) -> None:
        elbow_specs = (
            ("L", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", elbow_angle),
            ("R", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", elbow_angle),
        )
        knee_specs = (
            ("LK", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", knee_angle),
            ("RK", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", knee_angle),
        )
        for label, proximal, joint_name, distal, angle_fn in (*elbow_specs, *knee_specs):
            joint = find_landmark(landmarks, joint_name)
            angle = angle_fn(
                find_landmark(landmarks, proximal),
                joint,
                find_landmark(landmarks, distal),
                width,
                height,
            )
            if joint is None or angle is None:
                continue
            if not is_visible(joint, self.visibility_threshold):
                continue
            angle_text = format_angle(angle)
            if angle_text is None:
                continue
            x, y = _to_pixel(joint, width, height)
            cv2.putText(
                canvas,
                f"{label} {angle_text}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Desenha as conexões entre os landmarks
    def _draw_connections(
        self,
        canvas: np.ndarray,
        by_name: dict[str, LandmarkPoint],
        width: int,
        height: int,
    ) -> None:
        for start_name, end_name in self._connections:
            start = by_name.get(start_name)
            end = by_name.get(end_name)
            if start is None or end is None:
                continue
            if not is_visible(start, self.visibility_threshold):
                continue
            if not is_visible(end, self.visibility_threshold):
                continue
            cv2.line(
                canvas,
                _to_pixel(start, width, height),
                _to_pixel(end, width, height),
                (0, 255, 0),
                self.line_thickness,
            )

    # Desenha os pontos dos landmarks
    def _draw_points(
        self,
        canvas: np.ndarray,
        landmarks: list[LandmarkPoint],
        width: int,
        height: int,
    ) -> None:
        for landmark in landmarks:
            if not is_visible(landmark, self.visibility_threshold):
                continue
            cv2.circle(
                canvas,
                _to_pixel(landmark, width, height),
                self.point_radius,
                (0, 0, 255),
                thickness=-1,
            )

# Converte o landmark para pixels
def _to_pixel(landmark: LandmarkPoint, width: int, height: int) -> tuple[int, int]:
    x = int(round(landmark.x * width))
    y = int(round(landmark.y * height))
    return x, y
