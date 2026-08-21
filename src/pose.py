from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

LANDMARK_NAMES = {item.value: item.name for item in PoseLandmark}

# Detalhes de NOSE para um único ponto HEAD
FACE_LANDMARK_NAMES = frozenset(
    {
        "NOSE",
        "LEFT_EYE_INNER",
        "LEFT_EYE",
        "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER",
        "RIGHT_EYE",
        "RIGHT_EYE_OUTER",
        "LEFT_EAR",
        "RIGHT_EAR",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
    }
)

# Detalhes dos dedos para um único ponto HAND
HAND_TIP_LANDMARK_NAMES = frozenset(
    {
        "LEFT_PINKY",
        "RIGHT_PINKY",
        "LEFT_INDEX",
        "RIGHT_INDEX",
        "LEFT_THUMB",
        "RIGHT_THUMB",
    }
)

DROPPED_LANDMARK_NAMES = FACE_LANDMARK_NAMES | HAND_TIP_LANDMARK_NAMES


@dataclass(frozen=True)
class LandmarkPoint:
    # x/y/z normalizados (0–1); z é profundidade relativa.
    name: str
    x: float
    y: float
    z: float
    visibility: float | None

# Detecção de pose
class PoseDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        ensure_model(self.model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[LandmarkPoint] | None:
        # MediaPipe espera RGB; OpenCV entrega BGR.
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        pose = result.pose_landmarks[0]
        landmarks: list[LandmarkPoint] = []
        for index, landmark in enumerate(pose):
            landmarks.append(
                LandmarkPoint(
                    name=LANDMARK_NAMES.get(index, f"LANDMARK_{index}"),
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z) if landmark.z is not None else 0.0,
                    visibility=float(landmark.visibility)
                    if landmark.visibility is not None
                    else None,
                )
            )
        return simplify_landmarks(landmarks)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def ensure_model(model_path: Path) -> None:
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MODEL_URL, model_path)

# Encontra os landmarks pelo nome
def find_landmark(landmarks: list[LandmarkPoint], name: str) -> LandmarkPoint | None:
    for landmark in landmarks:
        if landmark.name == name:
            return landmark
    return None


def simplify_landmarks(landmarks: list[LandmarkPoint]) -> list[LandmarkPoint]:
    """Face → HEAD (NOSE); remove pontas dos dedos (mantém só o pulso)."""
    nose = find_landmark(landmarks, "NOSE")
    simplified: list[LandmarkPoint] = []
    if nose is not None:
        simplified.append(
            LandmarkPoint(
                name="HEAD",
                x=nose.x,
                y=nose.y,
                z=nose.z,
                visibility=nose.visibility,
            )
        )
    for landmark in landmarks:
        if landmark.name in DROPPED_LANDMARK_NAMES:
            continue
        simplified.append(landmark)
    return simplified
