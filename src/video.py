# Leitura e escrita de vídeo com OpenCV (separado da lógica de pose).

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    # Abre o MP4 de entrada, lê metadados e entrega os frames um a um.

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._capture: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def open(self) -> VideoInfo:
        if not self.path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {self.path}")

        self._capture = capture
        self.info = VideoInfo(
            path=self.path,
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return self.info

    def frames(self):
        if self._capture is None:
            raise RuntimeError("O vídeo precisa ser aberto com open() antes da leitura.")

        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoWriter:
    # Grava o MP4 de saída com o mesmo FPS e resolução da entrada.

    def __init__(self, path: Path, info: VideoInfo) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = info.fps if info.fps > 0 else 30.0
        # mp4v funciona no Windows; alguns players nativos preferem VLC.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            fps,
            (info.width, info.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Não foi possível criar o vídeo de saída: {self.path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
