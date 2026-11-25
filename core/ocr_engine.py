"""RapidOCR PP-OCRv6 multilingual detection and recognition wrapper."""

from __future__ import annotations

import logging
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .gpu import inspect_gpu
from .video_sampler import SampledFrame

if TYPE_CHECKING:
    from rapidocr import RapidOCR

LOGGER = logging.getLogger(__name__)
# PP-OCRv6 small routes all of these codes to the same multilingual weights.
# The route name "multi" itself is not accepted by RapidOCR 3.10.0.
LANGUAGES = {"auto": "ch", "ch": "ch", "vi": "vi", "ja": "japan", "en": "en"}


@dataclass(frozen=True)
class TextBox:
    points: tuple[tuple[float, float], ...]
    score: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(p[0] for p in self.points), min(p[1] for p in self.points),
            max(p[0] for p in self.points), max(p[1] for p in self.points),
        )


@dataclass(frozen=True)
class RecognizedText:
    text: str
    score: float
    box: TextBox


def _image(frame: SampledFrame | Image.Image) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame
    return Image.frombytes("RGB", (frame.width, frame.height), frame.rgb24)


class OCREngine:
    """One multilingual OCR model, with a fast detection-only path for ROI."""

    def __init__(self, language: str = "auto", *, prefer_gpu: bool = True, model_size: str = "small") -> None:
        if language not in LANGUAGES:
            raise ValueError(f"Ngôn ngữ không hỗ trợ: {language}")
        if model_size not in {"small", "medium"}:
            raise ValueError("model_size phải là small hoặc medium (tiny không hỗ trợ tiếng Nhật)")
        try:
            import rapidocr
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise RuntimeError("Cần cài rapidocr và onnxruntime/onnxruntime-gpu") from exc

        detector_model = Path(rapidocr.__file__).parent / "models" / f"PP-OCRv6_det_{model_size}.onnx"
        status = inspect_gpu(detector_model if detector_model.is_file() else None) if prefer_gpu else None
        self.using_gpu = bool(status and status.cuda_session_initialized)
        if prefer_gpu and not self.using_gpu:
            LOGGER.warning("OCR chạy trên CPU; không có phiên CUDA khả dụng.")
        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": "ch",
            "Det.model_type": ModelType(model_size),
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.limit_side_len": 1280,
            "Det.limit_type": "max",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LANGUAGES[language],
            "Rec.model_type": ModelType(model_size),
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "EngineConfig.onnxruntime.use_cuda": self.using_gpu,
            "Global.log_level": "error",
        }
        self._engine: RapidOCR = RapidOCR(params=params)
        self.language = language

    def detect(self, frame: SampledFrame | Image.Image) -> list[TextBox]:
        result = self._engine(_image(frame), use_det=True, use_cls=False, use_rec=False)
        if result.boxes is None:
            return []
        scores = result.scores or []
        return [
            TextBox(tuple((float(x), float(y)) for x, y in box), float(scores[i]) if i < len(scores) else 1.0)
            for i, box in enumerate(result.boxes)
        ]

    def recognize(self, frame: SampledFrame | Image.Image) -> list[RecognizedText]:
        result = self._engine(_image(frame), use_det=True, use_cls=False, use_rec=True)
        if result.boxes is None or result.txts is None:
            return []
        scores = result.scores or []
        return [
            RecognizedText(
                text.strip(), float(scores[i]) if i < len(scores) else 0.0,
                TextBox(tuple((float(x), float(y)) for x, y in box), float(scores[i]) if i < len(scores) else 0.0),
            )
            for i, (box, text) in enumerate(zip(result.boxes, result.txts)) if text.strip()
        ]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Thử OCR trên ảnh tĩnh")
    parser.add_argument("image", type=Path)
    parser.add_argument("--lang", choices=LANGUAGES, default="auto")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Không tìm thấy ảnh: {args.image}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for line in OCREngine(language=args.lang).recognize(Image.open(args.image).convert("RGB")):
        print(f"{line.score:.3f}\t{line.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
