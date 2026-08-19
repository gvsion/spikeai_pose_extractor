from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose import PoseDetector, find_landmark
from renderer import PoseRenderer
from video import VideoReader, VideoWriter

DEFAULT_INPUT = PROJECT_ROOT / "input" / "ataque_volei.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "ataque_volei_pose.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
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
    safe_fps = fps if fps > 0 else 30.0
    timestamp = int(round(frame_index * 1000.0 / safe_fps))
    if timestamp <= last_timestamp:
        timestamp = last_timestamp + 1
    return timestamp


def _print_sample(frame_index: int, landmarks) -> None:
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


def main() -> None:
    video_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT
    reader = VideoReader(video_path)
    info = reader.open()
    writer = VideoWriter(output_path, info)
    detector = PoseDetector(DEFAULT_MODEL)
    renderer = PoseRenderer()

    print(f"Vídeo: {info.path.name}")
    print(f"Resolução: {info.width}x{info.height}")
    print(f"FPS: {info.fps}")
    print(f"Frames: {info.frame_count}")

    processed = 0
    last_timestamp = -1
    sample_printed = False
    try:
        for frame in reader.frames():
            timestamp = _timestamp_ms(processed, info.fps, last_timestamp)
            last_timestamp = timestamp
            landmarks = detector.detect(frame, timestamp)
            if landmarks is not None and not sample_printed:
                _print_sample(processed, landmarks)
                sample_printed = True
            canvas = renderer.render(landmarks, info.width, info.height)
            writer.write(canvas)
            processed += 1
    finally:
        reader.close()
        writer.close()
        detector.close()

    print(f"Frames percorridos: {processed}")
    print(f"Saída: {output_path}")


if __name__ == "__main__":
    main()
