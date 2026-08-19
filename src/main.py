"""Ponto de entrada: lê o vídeo, detecta pose, desenha o skeleton e exporta dados."""

from pathlib import Path
import sys
import time

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export import frame_record, write_landmarks_json
from geometry import elbow_angle
from pose import PoseDetector, find_landmark
from renderer import PoseRenderer
from video import VideoReader, VideoWriter

DEFAULT_INPUT = PROJECT_ROOT / "input" / "ataque_volei.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "ataque_volei_pose.mp4"
DEFAULT_JSON = PROJECT_ROOT / "output" / "landmarks.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

# Recorte usado no print de diagnóstico (o vídeo desenha o esqueleto completo).
SAMPLE_LANDMARKS = (
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)


def _timestamp_ms(frame_index: int, fps: float, last_timestamp: int) -> int:
    """Timestamp crescente em ms — o PoseLandmarker em modo VIDEO exige isso."""
    safe_fps = fps if fps > 0 else 30.0
    timestamp = int(round(frame_index * 1000.0 / safe_fps))
    if timestamp <= last_timestamp:
        timestamp = last_timestamp + 1
    return timestamp


def _print_sample(frame_index: int, landmarks) -> None:
    """Mostra um frame no terminal para conferir coordenadas e ângulo do cotovelo."""
    print(f"Frame {frame_index}")
    print()
    for name in SAMPLE_LANDMARKS:
        landmark = find_landmark(landmarks, name)
        print(name)
        if landmark is None:
            print("ausente")
            print()
            continue
        visibility = landmark.visibility if landmark.visibility is not None else "n/a"
        print(f"x: {landmark.x:.4f}")
        print(f"y: {landmark.y:.4f}")
        print(f"z: {landmark.z:.4f}")
        print(f"visibility: {visibility}")
        print()

    left = elbow_angle(
        find_landmark(landmarks, "LEFT_SHOULDER"),
        find_landmark(landmarks, "LEFT_ELBOW"),
        find_landmark(landmarks, "LEFT_WRIST"),
    )
    right = elbow_angle(
        find_landmark(landmarks, "RIGHT_SHOULDER"),
        find_landmark(landmarks, "RIGHT_ELBOW"),
        find_landmark(landmarks, "RIGHT_WRIST"),
    )
    print("ELBOW_ANGLE")
    print(f"left: {left:.1f}°" if left is not None else "left: n/a")
    print(f"right: {right:.1f}°" if right is not None else "right: n/a")
    print()


def _print_stats(info, processed: int, with_pose: int, elapsed_s: float) -> None:
    without_pose = processed - with_pose
    rate = (with_pose / processed * 100) if processed else 0.0
    print(f"Vídeo: {info.path.name}")
    print(f"Resolução: {info.width}x{info.height}")
    print(f"FPS: {info.fps:.2f}")
    print(f"Frames processados: {processed}")
    print(f"Frames com pose detectada: {with_pose}")
    print(f"Frames sem detecção: {without_pose}")
    print(f"Taxa de detecção: {rate:.1f}%")
    print(f"Tempo de processamento: {elapsed_s:.1f}s")


def main() -> None:
    reader = VideoReader(DEFAULT_INPUT)
    info = reader.open()
    writer = VideoWriter(DEFAULT_OUTPUT, info)
    detector = PoseDetector(DEFAULT_MODEL)
    renderer = PoseRenderer()

    processed = 0
    with_pose = 0
    last_timestamp = -1
    sample_printed = False
    records: list[dict] = []
    started = time.perf_counter()
    try:
        for frame in reader.frames():
            timestamp = _timestamp_ms(processed, info.fps, last_timestamp)
            last_timestamp = timestamp
            landmarks = detector.detect(frame, timestamp)
            if landmarks is not None:
                with_pose += 1
                if not sample_printed:
                    _print_sample(processed, landmarks)
                    sample_printed = True
            records.append(frame_record(processed, landmarks))
            # Sempre um frame novo (preto + skeleton). Nunca reutiliza a imagem original.
            canvas = renderer.render(landmarks, info.width, info.height)
            writer.write(canvas)
            processed += 1
    finally:
        reader.close()
        writer.close()
        detector.close()

    write_landmarks_json(DEFAULT_JSON, records)
    elapsed = time.perf_counter() - started
    _print_stats(info, processed, with_pose, elapsed)
    print(f"Saída: {DEFAULT_OUTPUT}")
    print(f"JSON: {DEFAULT_JSON}")


if __name__ == "__main__":
    main()
