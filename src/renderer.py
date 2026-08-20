import numpy as np
import cv2
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

from geometry import elbow_angle, format_angle, is_visible
from pose import LandmarkPoint, find_landmark

DEFAULT_VISIBILITY_THRESHOLD = 0.5


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
        self._connections = PoseLandmarksConnections.POSE_LANDMARKS

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

        pixels = [_to_pixel(landmark, width, height) for landmark in landmarks]
        visible = [is_visible(landmark, self.visibility_threshold) for landmark in landmarks]
        self._draw_connections(canvas, pixels, visible)
        self._draw_points(canvas, pixels, visible)
        self._draw_elbow_angles(canvas, landmarks, width, height)
        return canvas

    def _draw_elbow_angles(
        self,
        canvas: np.ndarray,
        landmarks: list[LandmarkPoint],
        width: int,
        height: int,
    ) -> None:
        specs = (
            ("L", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            ("R", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        )
        for label, shoulder_name, elbow_name, wrist_name in specs:
            elbow = find_landmark(landmarks, elbow_name)
            angle = elbow_angle(
                find_landmark(landmarks, shoulder_name),
                elbow,
                find_landmark(landmarks, wrist_name),
            )
            if elbow is None or angle is None:
                continue
            if not is_visible(elbow, self.visibility_threshold):
                continue
            angle_text = format_angle(angle)
            if angle_text is None:
                continue
            x, y = _to_pixel(elbow, width, height)
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

    def _draw_connections(
        self,
        canvas: np.ndarray,
        pixels: list[tuple[int, int]],
        visible: list[bool],
    ) -> None:
        for connection in self._connections:
            if connection.start >= len(pixels) or connection.end >= len(pixels):
                continue
            if not visible[connection.start] or not visible[connection.end]:
                continue
            cv2.line(
                canvas,
                pixels[connection.start],
                pixels[connection.end],
                (0, 255, 0),
                self.line_thickness,
            )

    def _draw_points(
        self,
        canvas: np.ndarray,
        pixels: list[tuple[int, int]],
        visible: list[bool],
    ) -> None:
        for point, is_on in zip(pixels, visible):
            if not is_on:
                continue
            cv2.circle(canvas, point, self.point_radius, (0, 0, 255), thickness=-1)


def _to_pixel(landmark: LandmarkPoint, width: int, height: int) -> tuple[int, int]:
    x = int(round(landmark.x * width))
    y = int(round(landmark.y * height))
    return x, y
