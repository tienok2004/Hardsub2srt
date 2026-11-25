"""Adaptive, sparse-frame subtitle ROI detection with safe full-frame fallback."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from .ocr_engine import OCREngine, TextBox
from .video_sampler import SampledFrame, VideoMetadata, probe_video, sample_frame_at

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ROI:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Tọa độ ROI phải nằm trong [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI phải có chiều rộng và chiều cao dương")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (round(self.x_min * width), round(self.y_min * height),
                round(self.x_max * width), round(self.y_max * height))


FULL_FRAME = ROI(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ROIResult:
    roi: ROI
    mode: str  # manual, auto, full_frame
    support_frames: int
    frames_checked: int
    preview_path: Path
    message: str


class TextDetector(Protocol):
    def detect(self, frame: SampledFrame) -> list[TextBox]: ...


@dataclass
class _Cluster:
    boxes: list[tuple[float, float, float, float]]
    frame_ids: set[int]
    centers: list[tuple[float, float]]
    times: list[float]
    vertical: bool

    def accepts(self, box: tuple[float, float, float, float], vertical: bool) -> bool:
        if self.vertical != vertical:
            return False
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        reference_x = sum(point[0] for point in self.centers) / len(self.centers)
        reference_y = sum(point[1] for point in self.centers) / len(self.centers)
        return (abs(cx - reference_x) < 0.10 and abs(cy - reference_y) < 0.30) if vertical else (
            abs(cx - reference_x) < 0.13 and abs(cy - reference_y) < 0.04
        )

    def add(self, box: tuple[float, float, float, float], frame_id: int, timestamp: float) -> None:
        self.boxes.append(box)
        self.frame_ids.add(frame_id)
        self.centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        self.times.append(timestamp)


def _candidate(box: TextBox, width: int, height: int) -> tuple[tuple[float, float, float, float], bool] | None:
    x1, y1, x2, y2 = box.bounds
    if box.score < 0.35 or x2 <= x1 or y2 <= y1:
        return None
    rect = (max(0, x1 / width), max(0, y1 / height), min(1, x2 / width), min(1, y2 / height))
    vertical = (y2 - y1) > 1.25 * (x2 - x1)
    if (rect[3] - rect[1] if vertical else rect[2] - rect[0]) < 0.08:
        return None
    return rect, vertical


def _best_roi(
    observations: list[tuple[int, float, tuple[float, float, float, float], bool]],
    *,
    minimum_support_frames: int,
    padding_ratio: float,
) -> tuple[ROI, int] | None:
    clusters: list[_Cluster] = []
    for frame_id, timestamp, rect, vertical in observations:
        cluster = next((item for item in clusters if item.accepts(rect, vertical)), None)
        if cluster is None:
            cluster = _Cluster([], set(), [], [], vertical)
            clusters.append(cluster)
        cluster.add(rect, frame_id, timestamp)

    def looks_static(item: _Cluster) -> bool:
        if len(item.frame_ids) < 3 or max(item.times) - min(item.times) <= 15:
            return False
        # Fixed logos and URL overlays keep all four edges nearly unchanged.
        spans = [max(box[axis] for box in item.boxes) - min(box[axis] for box in item.boxes)
                 for axis in range(4)]
        if spans[0] < 0.08 and spans[2] < 0.08 and spans[1] < 0.04 and spans[3] < 0.04:
            return True
        center_x = sum(point[0] for point in item.centers) / len(item.centers)
        center_y = sum(point[1] for point in item.centers) / len(item.centers)
        probed_frames = max(frame_id for frame_id, *_ in observations) + 1
        return (len(item.frame_ids) / probed_frames >= 0.8
                and (center_x < 0.3 or center_x > 0.7)
                and (center_y < 0.15 or center_y > 0.85))

    eligible = [item for item in clusters
                if len(item.frame_ids) >= minimum_support_frames and not looks_static(item)]
    if not eligible:
        return None
    # Prefer repeated text near the horizontal center. Corner logos can be
    # frequent even when their edges vary slightly with compression.
    def rank(item: _Cluster) -> float:
        cx = sum(point[0] for point in item.centers) / len(item.centers)
        cy = sum(point[1] for point in item.centers) / len(item.centers)
        center_bonus = 0 if item.vertical else 5 * (1 - min(1.0, 2 * abs(cx - 0.5)))
        lower_bonus = 2 if not item.vertical and 0.55 <= cy <= 0.9 else 0
        return len(item.frame_ids) + center_bonus + lower_bonus

    best = max(eligible, key=rank)
    x1 = max(0.0, min(box[0] for box in best.boxes) - padding_ratio)
    y1 = max(0.0, min(box[1] for box in best.boxes) - padding_ratio)
    x2 = min(1.0, max(box[2] for box in best.boxes) + padding_ratio)
    y2 = min(1.0, max(box[3] for box in best.boxes) + padding_ratio)
    return ROI(x1, y1, x2, y2), len(best.frame_ids)


def _preview(frame: SampledFrame, roi: ROI, path: Path) -> None:
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb24)
    draw = ImageDraw.Draw(image)
    draw.rectangle(roi.pixels(frame.width, frame.height), outline=(0, 255, 0), width=max(2, frame.width // 300))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=90)


def detect_roi(
    video: str | Path,
    detector: TextDetector,
    *,
    manual: ROI | None = None,
    preview_path: str | Path = "roi_preview.jpg",
    sample_frames: int = 30,
    max_sample_frames: int = 90,
    probe_max_width: int = 960,
    minimum_support_frames: int = 2,
    padding_ratio: float = 0.03,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> ROIResult:
    """Use cheap detection probes; never guess a crop when evidence is weak."""
    if not 1 <= sample_frames <= max_sample_frames or minimum_support_frames < 1:
        raise ValueError("Số khung hình thăm dò không hợp lệ")
    if not 0 <= padding_ratio < 0.5 or probe_max_width <= 0:
        raise ValueError("padding_ratio hoặc probe_max_width không hợp lệ")
    info: VideoMetadata = probe_video(video, ffprobe=ffprobe)
    destination = Path(preview_path).expanduser().resolve()

    def read_at(timestamp: float) -> SampledFrame:
        # Container duration may extend beyond the final decodable frame.
        tail_margin = max(0.5, 2 / info.source_fps) if info.source_fps > 0 else 0.5
        safe_time = min(timestamp, max(0.0, info.duration_seconds - tail_margin))
        return sample_frame_at(info.path, safe_time,
                               max_width=probe_max_width, ffmpeg=ffmpeg, metadata=info)

    if manual is not None:
        frame = read_at(info.duration_seconds / 2)
        _preview(frame, manual, destination)
        return ROIResult(manual, "manual", 0, 1, destination, "Đã dùng ROI thủ công.")

    observations: list[tuple[int, float, tuple[float, float, float, float], bool]] = []
    seen_times: set[float] = set()
    preview_frame: SampledFrame | None = None
    checked = 0

    def probe(times: list[float], *, stop_when_supported: bool = False) -> tuple[ROI, int] | None:
        nonlocal checked, preview_frame
        for timestamp in times:
            key = round(timestamp, 3)
            if key in seen_times:
                continue
            seen_times.add(key)
            frame = read_at(timestamp)
            if preview_frame is None:
                preview_frame = frame
            for detected in detector.detect(frame):
                candidate = _candidate(detected, frame.width, frame.height)
                if candidate:
                    observations.append((checked, frame.timestamp_seconds, *candidate))
                    preview_frame = frame
            checked += 1
            if stop_when_supported:
                current = _best_roi(observations, minimum_support_frames=max(3, minimum_support_frames),
                                    padding_ratio=padding_ratio)
                if current is not None:
                    return current
        return _best_roi(observations, minimum_support_frames=minimum_support_frames,
                         padding_ratio=padding_ratio)

    primary = [info.duration_seconds * (0.05 + 0.90 * (i + 0.5) / sample_frames)
               for i in range(sample_frames)]
    found = probe(primary)
    if found is None and max_sample_frames > sample_frames:
        remaining = max_sample_frames - sample_frames
        # Interleaved points include the intro/outro and gaps between first-pass probes.
        extra = [info.duration_seconds * (i + 0.5) / remaining for i in range(remaining)]
        found = probe(extra, stop_when_supported=True)
    assert preview_frame is not None
    if found is None:
        LOGGER.warning("Không đủ bằng chứng để chọn ROI; sẽ OCR toàn khung hình.")
        roi, support, mode = FULL_FRAME, 0, "full_frame"
        message = "Không xác định được ROI đáng tin cậy; dùng toàn khung hình."
    else:
        roi, support = found
        mode = "auto"
        message = f"ROI tự động được hỗ trợ bởi {support} khung hình."
    _preview(preview_frame, roi, destination)
    return ROIResult(roi, mode, support, checked, destination, message)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tìm vùng phụ đề và xuất roi_preview.jpg")
    parser.add_argument("video", type=Path)
    parser.add_argument("--preview", type=Path, default=Path("roi_preview.jpg"))
    parser.add_argument("--manual", nargs=4, type=float, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manual = ROI(*args.manual) if args.manual else None
    engine = OCREngine() if manual is None else None
    result = detect_roi(args.video, engine, manual=manual, preview_path=args.preview,
                        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
    print(result.message)
    print(f"ROI: {result.roi}")
    print(f"Đã kiểm tra: {result.frames_checked} khung hình")
    print(f"Ảnh xem trước: {result.preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
