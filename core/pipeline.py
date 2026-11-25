"""Single-file and sequential batch orchestration, independent of the UI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm_refiner import LLMConfig, refine_subtitles
from .ocr_engine import OCREngine, RecognizedText
from .roi_detector import FULL_FRAME, ROI, ROIResult, detect_roi
from .srt_builder import BuilderConfig, build_subtitles, frame_from_ocr, write_srt
from .video_sampler import probe_video, sample_frames

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"})


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    language: str = "auto"
    fps: float = 3.0
    recursive: bool = False
    preserve_subfolders: bool = False
    overwrite: bool = False
    auto_roi: bool = True
    manual_roi: ROI | None = None
    roi_sample_frames: int = 30
    roi_max_sample_frames: int = 90
    roi_probe_max_width: int = 960
    roi_minimum_support_frames: int = 2
    roi_padding_ratio: float = 0.03
    prefer_gpu: bool = True
    ocr_model_size: str = "small"
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.fps > 10:
            raise ValueError("FPS lấy mẫu phải nằm trong (0, 10]")
        if self.language not in {"auto", "ch", "vi", "ja", "en"}:
            raise ValueError("Ngôn ngữ OCR không hợp lệ")


@dataclass(frozen=True)
class ProgressEvent:
    video: Path
    phase: str
    processed_frames: int = 0
    duration_seconds: float = 0.0
    processing_fps: float = 0.0
    roi: ROI | None = None
    message: str = ""
    preview_path: Path | None = None


@dataclass(frozen=True)
class FileResult:
    video: Path
    output: Path
    status: str
    subtitle_count: int = 0
    processed_frames: int = 0
    elapsed_seconds: float = 0.0
    roi: ROI | None = None
    message: str = ""


@dataclass(frozen=True)
class BatchSummary:
    files: tuple[FileResult, ...]

    @property
    def completed(self) -> int:
        return sum(result.status == "completed" for result in self.files)

    @property
    def skipped(self) -> int:
        return sum(result.status.startswith("skipped") for result in self.files)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.files)


ProgressCallback = Callable[[ProgressEvent], None]


def _near_roi_anchor(lines: list[RecognizedText], roi: ROI, frame_width: int) -> list[RecognizedText]:
    """Keep subtitle boxes near the cluster center, excluding adjacent logos."""
    anchor_x = (roi.x_min + roi.x_max) / 2
    roi_width = roi.x_max - roi.x_min
    kept = []
    for line in lines:
        x1, _, x2, _ = line.box.bounds
        x_center = roi.x_min + ((x1 + x2) / 2 / frame_width) * roi_width
        if abs(x_center - anchor_x) <= 0.18:
            kept.append(line)
    return kept


def _videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.casefold() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Định dạng video không hỗ trợ: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy đầu vào: {input_path}")
    candidates = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def _output_path(video: Path, root: Path, config: PipelineConfig) -> Path:
    if config.preserve_subfolders and root.is_dir():
        return config.output_dir / video.relative_to(root).with_suffix(".srt")
    return config.output_dir / f"{video.stem}.srt"


def _preview_path(output: Path) -> Path:
    return output.with_name(output.stem + "_roi_preview.jpg")


def _process_one(
    video: Path, output: Path, config: PipelineConfig,
    engine: OCREngine, callback: ProgressCallback | None,
) -> FileResult:
    started = time.perf_counter()
    info = probe_video(video, ffprobe=config.ffprobe)
    if callback:
        callback(ProgressEvent(video, "roi", duration_seconds=info.duration_seconds))
    selected_manual = config.manual_roi if config.auto_roi else (config.manual_roi or FULL_FRAME)
    detected: ROIResult = detect_roi(
        video, engine, manual=selected_manual, preview_path=_preview_path(output),
        sample_frames=config.roi_sample_frames,
        max_sample_frames=config.roi_max_sample_frames,
        probe_max_width=config.roi_probe_max_width,
        minimum_support_frames=config.roi_minimum_support_frames,
        padding_ratio=config.roi_padding_ratio,
        ffmpeg=config.ffmpeg, ffprobe=config.ffprobe,
    )
    roi = detected.roi
    if callback:
        callback(ProgressEvent(video, "ocr", duration_seconds=info.duration_seconds,
                               roi=roi, message=detected.message,
                               preview_path=detected.preview_path))
    crop = None if roi == FULL_FRAME else roi.pixels(info.width, info.height)
    observations = []
    count = 0
    ocr_started = time.perf_counter()
    for frame in sample_frames(video, fps=config.fps, ffmpeg=config.ffmpeg,
                               metadata=info, crop=crop):
        recognized = engine.recognize(frame)
        if detected.mode == "auto" and crop is not None:
            recognized = _near_roi_anchor(recognized, roi, frame.width)
        observations.append(frame_from_ocr(frame.timestamp_seconds, frame.width, frame.height, recognized))
        count += 1
        if callback and (count == 1 or count % 10 == 0):
            elapsed = max(1e-9, time.perf_counter() - ocr_started)
            callback(ProgressEvent(video, "ocr", count, info.duration_seconds,
                                   count / elapsed, roi))
    entries = build_subtitles(observations, config=config.builder, sample_fps=config.fps)
    entries = refine_subtitles(entries, config.llm)
    write_srt(entries, output)
    elapsed = time.perf_counter() - started
    if callback:
        callback(ProgressEvent(video, "done", count, info.duration_seconds,
                               count / max(time.perf_counter() - ocr_started, 1e-9), roi,
                               f"{len(entries)} phụ đề", detected.preview_path))
    return FileResult(video, output, "completed", len(entries), count, elapsed, roi,
                      f"{len(entries)} phụ đề; {detected.message}")


def run_batch(
    config: PipelineConfig, *, callback: ProgressCallback | None = None,
    engine: OCREngine | None = None,
) -> BatchSummary:
    """Process videos sequentially, with deterministic collision and resume rules."""
    root = config.input_path.expanduser().resolve()
    videos = _videos(root, config.recursive)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = PipelineConfig(**{**vars(config), "input_path": root, "output_dir": output_dir})
    results: list[FileResult] = []
    destinations: set[str] = set()
    for video in videos:
        output = _output_path(video, root, effective)
        key = str(output).casefold()
        if key in destinations:
            message = f"Tên đầu ra trùng với video đã xét trước: {output.name}"
            LOGGER.warning("Bỏ qua %s: %s", video, message)
            results.append(FileResult(video, output, "skipped_collision", message=message))
            continue
        destinations.add(key)
        if output.is_file() and not effective.overwrite:
            results.append(FileResult(video, output, "skipped_existing", message="SRT đã tồn tại"))
            continue
        try:
            active_engine = engine or OCREngine(effective.language, prefer_gpu=effective.prefer_gpu,
                                               model_size=effective.ocr_model_size)
            engine = active_engine
            results.append(_process_one(video, output, effective, active_engine, callback))
        except Exception as exc:
            LOGGER.error("Không xử lý được %s: %s", video, exc)
            if callback:
                callback(ProgressEvent(video, "failed", message=str(exc)))
            results.append(FileResult(video, output, "failed", message=str(exc)))
    return BatchSummary(tuple(results))
