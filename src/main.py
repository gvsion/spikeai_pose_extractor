from pathlib import Path
import sys
import time

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from export import angle_row, landmark_rows, write_angles_csv, write_landmarks_csv
from geometry import elbow_angle, format_angle, knee_angle
from pose import PoseDetector, find_landmark
from renderer import PoseRenderer
from video import VideoReader, VideoWriter

DEFAULT_INPUT = PROJECT_ROOT / "input" / "ataque_volei.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "ataque_volei_pose.mp4"
DEFAULT_OVERLAY = PROJECT_ROOT / "output" / "original_pose.mp4"
DEFAULT_LANDMARKS_CSV = PROJECT_ROOT / "output" / "landmarks.csv"
DEFAULT_ANGLES_CSV = PROJECT_ROOT / "output" / "angles.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

# Dividindo os landmarks em grupos para melhor visualização
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

# Função para calcular o timestamp em milissegundos
def _timestamp_ms(frame_index: int, fps: float, last_timestamp: int) -> int:
    safe_fps = fps if fps > 0 else 30.0
    timestamp = int(round(frame_index * 1000.0 / safe_fps))
    if timestamp <= last_timestamp:
        timestamp = last_timestamp + 1
    return timestamp

# Função para imprimir os landmarks e o ângulo do cotovelo e joelho
def _print_sample(frame_index: int, landmarks, width: int, height: int) -> None:
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
        print(f"visibility: {visibility if isinstance(visibility, str) else f'{visibility:.4f}'}")
        print()

    left = elbow_angle(
        find_landmark(landmarks, "LEFT_SHOULDER"),
        find_landmark(landmarks, "LEFT_ELBOW"),
        find_landmark(landmarks, "LEFT_WRIST"),
        width,
        height,
    )
    right = elbow_angle(
        find_landmark(landmarks, "RIGHT_SHOULDER"),
        find_landmark(landmarks, "RIGHT_ELBOW"),
        find_landmark(landmarks, "RIGHT_WRIST"),
        width,
        height,
    )
    print("ELBOW_ANGLE")
    left_text = format_angle(left)
    right_text = format_angle(right)
    print(f"left: {left_text}°" if left_text is not None else "left: n/a")
    print(f"right: {right_text}°" if right_text is not None else "right: n/a")
    print()

    left_knee = knee_angle(
        find_landmark(landmarks, "LEFT_HIP"),
        find_landmark(landmarks, "LEFT_KNEE"),
        find_landmark(landmarks, "LEFT_ANKLE"),
        width,
        height,
    )
    right_knee = knee_angle(
        find_landmark(landmarks, "RIGHT_HIP"),
        find_landmark(landmarks, "RIGHT_KNEE"),
        find_landmark(landmarks, "RIGHT_ANKLE"),
        width,
        height,
    )
    print("KNEE_ANGLE")
    left_knee_text = format_angle(left_knee)
    right_knee_text = format_angle(right_knee)
    print(f"left: {left_knee_text}°" if left_knee_text is not None else "left: n/a")
    print(f"right: {right_knee_text}°" if right_knee_text is not None else "right: n/a")
    print()

# Função para imprimir as estatísticas do vídeo
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
    only_writer = VideoWriter(DEFAULT_OUTPUT, info)
    overlay_writer = VideoWriter(DEFAULT_OVERLAY, info)
    detector = PoseDetector(DEFAULT_MODEL)
    renderer = PoseRenderer()

    processed = 0
    with_pose = 0
    last_timestamp = -1
    sample_printed = False
    landmark_records: list[dict] = []
    angle_records: list[dict] = []
    started = time.perf_counter()
    try:
        for frame in reader.frames():
            timestamp = _timestamp_ms(processed, info.fps, last_timestamp)
            last_timestamp = timestamp
            landmarks = detector.detect(frame, timestamp)
            if landmarks is not None:
                with_pose += 1
                if not sample_printed:
                    _print_sample(processed, landmarks, info.width, info.height)
                    sample_printed = True
            landmark_records.extend(landmark_rows(processed, landmarks))
            angle_records.append(angle_row(processed, landmarks, info.width, info.height))
            only_pose = renderer.render(landmarks, info.width, info.height)
            original_pose = renderer.render(
                landmarks, info.width, info.height, base_frame=frame
            )
            only_writer.write(only_pose)
            overlay_writer.write(original_pose)
            processed += 1
    finally:
        reader.close()
        only_writer.close()
        overlay_writer.close()
        detector.close()
        
    # Escreve os landmarks e os ângulos em arquivos CSV
    write_landmarks_csv(DEFAULT_LANDMARKS_CSV, landmark_records)
    write_angles_csv(DEFAULT_ANGLES_CSV, angle_records)
    elapsed = time.perf_counter() - started
    _print_stats(info, processed, with_pose, elapsed)
    print(f"Somente pose: {DEFAULT_OUTPUT}")
    print(f"Original + pose: {DEFAULT_OVERLAY}")
    print(f"Landmarks CSV: {DEFAULT_LANDMARKS_CSV}")
    print(f"Angles CSV: {DEFAULT_ANGLES_CSV}")


if __name__ == "__main__":
    main()
