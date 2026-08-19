"""Desenha o skeleton em um frame preto. O frame original nunca entra na saída."""

import numpy as np
import cv2
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

from geometry import is_visible
from pose import LandmarkPoint

DEFAULT_VISIBILITY_THRESHOLD = 0.5


class PoseRenderer:
    """Pontos (vermelho) + conexões oficiais do MediaPipe (verde)."""

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
    ) -> np.ndarray:
        canvas = self.empty_frame(width, height)
        if not landmarks:
            return canvas

        pixels = [_to_pixel(landmark, width, height) for landmark in landmarks]
        visible = [is_visible(landmark, self.visibility_threshold) for landmark in landmarks]
        self._draw_connections(canvas, pixels, visible)
        self._draw_points(canvas, pixels, visible)
        return canvas

    def _draw_connections(
        self,
        canvas: np.ndarray,
        pixels: list[tuple[int, int]],
        visible: list[bool],
    ) -> None:
        for connection in self._connections:
            if connection.start >= len(pixels) or connection.end >= len(pixels):
                continue
            # Só liga dois pontos se ambos passaram no filtro de visibility.
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
    """Converte coordenada normalizada (0–1) para pixel do frame."""
    x = int(round(landmark.x * width))
    y = int(round(landmark.y * height))
    return x, y
