import numpy as np
import cv2
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

from pose import LandmarkPoint


class PoseRenderer:
    """Desenha landmarks e conexões em um frame vazio (nunca usa o frame original)."""

    def __init__(self, point_radius: int = 4, line_thickness: int = 2) -> None:
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self._connections = PoseLandmarksConnections.POSE_LANDMARKS

    def empty_frame(self, width: int, height: int) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def render(
        self,
        landmarks: list[LandmarkPoint] | None,
        width: int,
        height: int,
    ) -> np.ndarray:
        canvas = self.empty_frame(width, height)
        if not landmarks:
            return canvas

        pixels = [_to_pixel(landmark, width, height) for landmark in landmarks]
        self._draw_connections(canvas, pixels)
        self._draw_points(canvas, pixels)
        return canvas

    def _draw_connections(self, canvas: np.ndarray, pixels: list[tuple[int, int]]) -> None:
        for connection in self._connections:
            if connection.start >= len(pixels) or connection.end >= len(pixels):
                continue
            start = pixels[connection.start]
            end = pixels[connection.end]
            cv2.line(canvas, start, end, (0, 255, 0), self.line_thickness)

    def _draw_points(self, canvas: np.ndarray, pixels: list[tuple[int, int]]) -> None:
        for point in pixels:
            cv2.circle(canvas, point, self.point_radius, (0, 0, 255), thickness=-1)


def _to_pixel(landmark: LandmarkPoint, width: int, height: int) -> tuple[int, int]:
    x = int(round(landmark.x * width))
    y = int(round(landmark.y * height))
    return x, y
