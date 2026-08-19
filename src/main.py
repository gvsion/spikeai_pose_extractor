from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from video import VideoReader

DEFAULT_INPUT = PROJECT_ROOT / "input" / "ataque.mp4"


def main() -> None:
    video_path = DEFAULT_INPUT
    reader = VideoReader(video_path)
    info = reader.open()

    print(f"Vídeo: {info.path.name}")
    print(f"Resolução: {info.width}x{info.height}")
    print(f"FPS: {info.fps}")
    print(f"Frames: {info.frame_count}")

    processed = 0
    try:
        for _frame in reader.frames():
            processed += 1
    finally:
        reader.close()

    print(f"Frames percorridos: {processed}")


if __name__ == "__main__":
    main()
