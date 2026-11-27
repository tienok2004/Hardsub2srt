"""Single-file and sequential batch orchestration, independent of the UI."""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .captions import (TOP_TAG, CaptionConfig, CaptionFrame, CaptionText, ScreenTextConfig,
                       angle_of, build_captions, build_screen_text, is_corner_overlay,
                       merge_entries)
from .frame_change import FrameChangeDetector
from .language_split import LANGUAGE_CODES, split_entries, split_observations
from .llm_refiner import LLMConfig, refine_subtitles
from .ocr_engine import OCREngine, RecognizedText
from .roi_detector import FULL_FRAME, ROI, ROIResult, detect_roi
from .srt_builder import (BuilderConfig, FrameObservation, SubtitleEntry, build_subtitles,
                          frame_from_ocr, remove_watermarks, write_srt)
from .video_sampler import SampledFrame, probe_video, sample_frames

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"})
# merged: captions in the same SRT (top-tagged); separate: <name>.captions.srt; off: dialogue only.
CAPTION_OUTPUTS = ("merged", "separate", "off")


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
    skip_unchanged_frames: bool = True
    captions_output: str = "merged"
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    screen_text: ScreenTextConfig = field(default_factory=ScreenTextConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.fps > 10:
            raise ValueError("FPS lấy mẫu phải nằm trong (0, 10]")
        if self.language not in {"auto", "ch", "vi", "ja", "en"}:
            raise ValueError("Ngôn ngữ OCR không hợp lệ")
        if self.captions_output not in CAPTION_OUTPUTS:
            raise ValueError(f"captions_output phải là một trong {CAPTION_OUTPUTS}")


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
    eta_seconds: float | None = None  # time left for this video; None until measurable


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
    outputs: tuple[Path, ...] = ()


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

    @property
    def stopped(self) -> bool:
        return any(result.status == "stopped" for result in self.files)


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


def partial_output_path(output: Path) -> Path:
    """Where a stopped run saves what it read so far.

    A separate name keeps resume working: the next run does not mistake a
    half-finished file for a completed ``.srt`` and will process the video again.
    """
    return output.with_name(output.stem + ".partial.srt")


def _language_output_path(output: Path, language: str) -> Path:
    return output.with_name(f"{output.stem}.{language}.srt")


def _existing_language_outputs(output: Path) -> tuple[Path, ...]:
    return tuple(path for language in LANGUAGE_CODES
                 if (path := _language_output_path(output, language)).is_file())


def _caption_zones(roi: ROI, zones: tuple[tuple[float, float, float, float], ...]) -> list[ROI]:
    """Caption search areas, with the dialogue band cut out of them."""
    result = []
    for x1, y1, x2, y2 in zones:
        if y2 > roi.y_min - 0.01 and y1 < roi.y_max + 0.01:
            above = (y1, min(y2, roi.y_min - 0.01))
            below = (max(y1, roi.y_max + 0.01), y2)
            y1, y2 = max(above, below, key=lambda part: part[1] - part[0])
        if y2 - y1 >= 0.05:
            result.append(ROI(x1, y1, x2, y2))
    return result


def _without_screen_region(
    frames: list[FrameObservation], entries: list[SubtitleEntry], zones: list[ROI],
) -> list[FrameObservation]:
    """When auto ROI falls back to full frame, keep accepted phone text out of dialogue."""
    if not entries:
        return frames
    result = []
    for frame in frames:
        time_ms = round(frame.timestamp_seconds * 1000)
        if not any(entry.start_ms <= time_ms < entry.end_ms for entry in entries):
            result.append(frame)
            continue
        texts = tuple(text for text in frame.texts if text.box is None or (
            not any(zone.x_min <= (text.box[0] + text.box[2]) / 2 <= zone.x_max
                    and zone.y_min <= (text.box[1] + text.box[3]) / 2 <= zone.y_max
                    for zone in zones)
            and not (is_corner_overlay(text.box)
                     and sum(ch.isalnum() for ch in text.text) <= 6)))
        result.append(FrameObservation(frame.timestamp_seconds, texts))
    return result


def _crop(frame: SampledFrame, box: tuple[int, int, int, int]) -> SampledFrame:
    x1, y1, x2, y2 = box
    pixels = np.frombuffer(frame.rgb24, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    part = np.ascontiguousarray(pixels[y1:y2, x1:x2])
    return SampledFrame(frame.timestamp_seconds, x2 - x1, y2 - y1, part.tobytes())


def _as_caption(line: RecognizedText, origin: tuple[int, int], width: int, height: int) -> CaptionText:
    """Move a line read inside a crop into full-frame normalized coordinates."""
    x1, y1, x2, y2 = line.box.bounds
    ox, oy = origin
    vertical = (y2 - y1) >= 1.35 * (x2 - x1)
    return CaptionText(line.text, line.score,
                       ((ox + x1) / width, (oy + y1) / height, (ox + x2) / width, (oy + y2) / height),
                       angle_of(line.box.points), vertical)


def _write_outputs(
    dialogue: dict[str, list[SubtitleEntry]], captions: dict[str, list[SubtitleEntry]],
    destination: Path, mode: str, primary: str,
) -> list[Path]:
    languages = {language for language in set(dialogue) | set(captions)
                 if dialogue.get(language) or (mode != "off" and captions.get(language))}
    written: list[Path] = []
    for language in sorted(languages or {primary}, key=lambda code: (code != primary, code)):
        path = _language_output_path(destination, language)
        words = dialogue.get(language, [])
        notes = captions.get(language, [])
        if mode == "merged":
            written.append(write_srt(merge_entries(words, notes), path))
        else:
            written.append(write_srt(words, path))
            if mode == "separate" and notes:
                written.append(write_srt(notes, path.with_name(path.stem + ".captions.srt")))
    return written


def format_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class EtaEstimator:
    """Estimate time left from the recent processing rate.

    Speed changes a lot within one video: frames that skip OCR are ~10x faster
    than frames that need it, so a dialogue-heavy scene is much slower than a
    quiet one. A sliding window follows those changes, while the whole-run
    average is used until the window has enough data.
    """

    def __init__(self, total_frames: int, *, window_seconds: float = 30.0, min_seconds: float = 3.0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.total_frames = max(0, total_frames)
        self.window_seconds = window_seconds
        self.min_seconds = min_seconds
        self._clock = clock
        self._started = clock()
        self._samples: deque[tuple[float, int]] = deque([(self._started, 0)])
        self._baseline_set = False
        self._base_frames = 0

    def update(self, done_frames: int) -> float | None:
        now = self._clock()
        if not self._baseline_set and done_frames > 0:
            # The first OCR call carries one-off GPU warm-up (~1 s); measuring
            # from after it keeps the early estimate from being inflated.
            self._baseline_set = True
            self._started = now - 1e-9
            self._samples = deque([(now, done_frames)])
            self._base_frames = done_frames
            return None
        self._samples.append((now, done_frames))
        while len(self._samples) > 2 and now - self._samples[1][0] >= self.window_seconds:
            self._samples.popleft()
        if now - self._started < self.min_seconds or done_frames <= 0:
            return None
        old_time, old_frames = self._samples[0]
        if now - old_time >= self.min_seconds and done_frames > old_frames:
            rate = (done_frames - old_frames) / (now - old_time)
        else:
            rate = (done_frames - self._base_frames) / (now - self._started)
        if rate <= 0:
            return None
        return max(0, self.total_frames - done_frames) / rate


def _process_one(
    video: Path, output: Path, config: PipelineConfig,
    engine: OCREngine, callback: ProgressCallback | None,
    stop_event: threading.Event | None = None,
) -> FileResult:
    started = time.perf_counter()

    def stop_requested() -> bool:
        return stop_event is not None and stop_event.is_set()

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
    if stop_requested():
        if callback:
            callback(ProgressEvent(video, "stopped", duration_seconds=info.duration_seconds,
                                   roi=roi, message="Đã dừng trước khi OCR; chưa có phụ đề để lưu"))
        return FileResult(video, output, "stopped", 0, 0, time.perf_counter() - started, roi,
                          "Đã dừng trước khi OCR; không ghi tệp")
    if callback:
        callback(ProgressEvent(video, "ocr", duration_seconds=info.duration_seconds,
                               roi=roi, message=detected.message,
                               preview_path=detected.preview_path))
    crop = None if roi == FULL_FRAME else roi.pixels(info.width, info.height)
    # Text outside the dialogue ROI needs whole frames. Crop the dialogue and
    # extra OCR areas here; full-frame ROI already reads everything as dialogue.
    read_captions = config.captions_output != "off" and crop is not None
    read_screen = config.captions_output != "off" and config.screen_text.enabled
    read_whole_frame = read_captions or read_screen
    zones = [zone.pixels(info.width, info.height)
             for zone in _caption_zones(roi, config.captions.zones)] if read_captions else []
    screen_areas = (_caption_zones(roi, config.screen_text.zones) if crop is not None
                    else [ROI(*zone) for zone in config.screen_text.zones]) if read_screen else []
    screen_zones = [zone.pixels(info.width, info.height) for zone in screen_areas]
    zone_detectors = [FrameChangeDetector() if config.skip_unchanged_frames else None for _ in zones]
    zone_lines: list[list[CaptionText]] = [[] for _ in zones]
    screen_detectors = [FrameChangeDetector() if config.skip_unchanged_frames else None for _ in screen_zones]
    screen_lines: list[list[CaptionText]] = [[] for _ in screen_zones]
    caption_frames: list[CaptionFrame] = []
    screen_frames: list[CaptionFrame] = []
    dialogue_heights: list[float] = []
    off_anchor: list[CaptionText] = []
    observations = []
    count = 0
    reused = 0
    change_detector = FrameChangeDetector() if config.skip_unchanged_frames else None
    recognized: list[RecognizedText] = []
    ocr_started = time.perf_counter()
    eta = EtaEstimator(math.ceil(info.duration_seconds * config.fps))
    stopped = False
    frames = sample_frames(video, fps=config.fps, ffmpeg=config.ffmpeg, metadata=info,
                           crop=None if read_whole_frame else crop)
    with closing(frames):  # Closing the generator stops FFmpeg right away.
        for whole in frames:
            if stop_requested():
                stopped = True
                break
            frame = _crop(whole, crop) if read_whole_frame and crop is not None else whole
            if change_detector is None or change_detector.changed(frame):
                recognized = engine.recognize(frame)
                if detected.mode == "auto" and crop is not None:
                    kept = _near_roi_anchor(recognized, roi, frame.width)
                    # Off-centre text in the dialogue band (e.g. a caption at the
                    # bottom-left) is checked as a caption instead of dropped.
                    off_anchor = [_as_caption(line, crop[:2], info.width, info.height)
                                  for line in recognized if line not in kept] if read_captions else []
                    recognized = kept
                # Reference font size for captions; only confident, real words, so
                # stray boxes from the picture do not drag the median down.
                dialogue_heights.extend((line.box.bounds[3] - line.box.bounds[1]) / info.height
                                        for line in recognized
                                        if line.score >= 0.8 and sum(ch.isalpha() for ch in line.text) >= 2)
            else:
                reused += 1  # Subtitle area unchanged: same text, new timestamp.
            observations.append(frame_from_ocr(frame.timestamp_seconds, frame.width, frame.height, recognized))
            if read_captions:
                for index, (box, detector) in enumerate(zip(zones, zone_detectors)):
                    zone_frame = _crop(whole, box)
                    if detector is None or detector.changed(zone_frame):
                        zone_lines[index] = [_as_caption(line, box[:2], info.width, info.height)
                                             for line in engine.recognize(zone_frame)]
                caption_frames.append(CaptionFrame(
                    whole.timestamp_seconds,
                    tuple(off_anchor) + tuple(text for lines in zone_lines for text in lines)))
            if read_screen:
                for index, (box, detector) in enumerate(zip(screen_zones, screen_detectors)):
                    zone_frame = _crop(whole, box)
                    if detector is None or detector.changed(zone_frame):
                        screen_lines[index] = [_as_caption(line, box[:2], info.width, info.height)
                                               for line in engine.recognize(zone_frame)]
                screen_frames.append(CaptionFrame(
                    whole.timestamp_seconds, tuple(text for lines in screen_lines for text in lines)))
            count += 1
            if callback and (count == 1 or count % 10 == 0):
                elapsed = max(1e-9, time.perf_counter() - ocr_started)
                callback(ProgressEvent(video, "ocr", count, info.duration_seconds,
                                       count / elapsed, roi, eta_seconds=eta.update(count)))
    caption_entries, caption_rejects = build_captions(
        caption_frames, interval=1 / config.fps, config=config.captions,
        dialogue_height=statistics.median(dialogue_heights) if dialogue_heights else None,
        watermark_terms=config.builder.watermark_terms,
    ) if read_captions else ([], {})
    screen_entries, screen_rejects = build_screen_text(
        screen_frames, interval=1 / config.fps, config=config.screen_text,
        ad_terms=config.captions.ad_terms, watermark_terms=config.builder.watermark_terms,
    ) if read_screen else ([], {})
    if screen_entries:
        # The upper edge of the screen area can overlap an editorial caption zone.
        caption_entries = [entry for entry in caption_entries if not any(
            entry.text.removeprefix(TOP_TAG).strip(" ()").casefold() == screen.text.strip(" ()").casefold()
            and entry.start_ms < screen.end_ms and screen.start_ms < entry.end_ms
            for screen in screen_entries)]
    if crop is None:
        observations = _without_screen_region(observations, screen_entries, screen_areas)
    observations = remove_watermarks(observations, config=config.builder, sample_fps=config.fps)
    language_frames, primary = split_observations(observations, selected_language=config.language)
    dialogue_by_language = {
        language: build_subtitles(frames, config=config.builder, sample_fps=config.fps)
        for language, frames in language_frames.items()
    }
    if caption_rejects:
        LOGGER.info("%s: bỏ qua chữ ngoài phụ đề: %s", video.name,
                    ", ".join(f"{reason} ×{n}" for reason, n in caption_rejects.items()))
    if screen_rejects:
        LOGGER.info("%s: bỏ qua chữ vùng màn hình: %s", video.name,
                    ", ".join(f"{reason} ×{n}" for reason, n in screen_rejects.items()))
    caption_entries.extend(screen_entries)
    captions_by_language = split_entries(caption_entries, primary=primary,
                                         selected_language=config.language)
    entries = [entry for group in dialogue_by_language.values() for entry in group]
    caption_note = f" + {len(caption_entries)} chữ trên màn hình/chú thích" if read_whole_frame else ""
    if stopped:
        # Save what was read so far; skip the optional LLM pass so stopping stays fast.
        written = tuple(_write_outputs(dialogue_by_language, captions_by_language, output,
                                       config.captions_output, primary))
        partial_paths = tuple(partial_output_path(path) for path in written)
        for path, partial in zip(written, partial_paths):
            path.replace(partial)
        scanned = count / config.fps
        message = (f"Đã dừng ở {format_clock(scanned)}/{format_clock(info.duration_seconds)}; "
                   f"lưu {len(entries)} phụ đề{caption_note} vào "
                   f"{', '.join(path.name for path in partial_paths)}")
        if callback:
            callback(ProgressEvent(video, "stopped", count, info.duration_seconds,
                                   count / max(time.perf_counter() - ocr_started, 1e-9), roi,
                                   message, detected.preview_path))
        return FileResult(video, partial_paths[0], "stopped", len(entries), count,
                          time.perf_counter() - started, roi, message, partial_paths)
    dialogue_by_language = {language: refine_subtitles(words, config.llm)
                            for language, words in dialogue_by_language.items()}
    written = tuple(_write_outputs(dialogue_by_language, captions_by_language, output,
                                   config.captions_output, primary))
    elapsed = time.perf_counter() - started
    language_note = ", ".join(path.name for path in written)
    if callback:
        callback(ProgressEvent(video, "done", count, info.duration_seconds,
                               count / max(time.perf_counter() - ocr_started, 1e-9), roi,
                               f"{len(entries)} phụ đề{caption_note}; {language_note}; "
                               f"OCR {count - reused}/{count} khung hình",
                               detected.preview_path))
    return FileResult(video, written[0], "completed", len(entries), count, elapsed, roi,
                      f"{len(entries)} phụ đề{caption_note}; {language_note}; {detected.message}", written)


def run_batch(
    config: PipelineConfig, *, callback: ProgressCallback | None = None,
    engine: OCREngine | None = None, stop_event: threading.Event | None = None,
) -> BatchSummary:
    """Process videos sequentially, with deterministic collision and resume rules.

    Setting ``stop_event`` stops at the next frame: the current video's subtitles
    so far are saved with ``partial_output_path`` and remaining videos are left.
    """
    root = config.input_path.expanduser().resolve()
    videos = _videos(root, config.recursive)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = PipelineConfig(**{**vars(config), "input_path": root, "output_dir": output_dir})
    results: list[FileResult] = []
    destinations: set[str] = set()
    for video in videos:
        if stop_event is not None and stop_event.is_set():
            break
        output = _output_path(video, root, effective)
        key = str(output).casefold()
        if key in destinations:
            message = f"Tên đầu ra trùng với video đã xét trước: {output.name}"
            LOGGER.warning("Bỏ qua %s: %s", video, message)
            results.append(FileResult(video, output, "skipped_collision", message=message))
            continue
        destinations.add(key)
        existing = _existing_language_outputs(output)
        if existing and not effective.overwrite:
            results.append(FileResult(video, existing[0], "skipped_existing",
                                      message="SRT theo ngôn ngữ đã tồn tại", outputs=existing))
            continue
        try:
            active_engine = engine or OCREngine(effective.language, prefer_gpu=effective.prefer_gpu,
                                               model_size=effective.ocr_model_size)
            engine = active_engine
            results.append(_process_one(video, output, effective, active_engine, callback, stop_event))
        except Exception as exc:
            LOGGER.error("Không xử lý được %s: %s", video, exc)
            if callback:
                callback(ProgressEvent(video, "failed", message=str(exc)))
            results.append(FileResult(video, output, "failed", message=str(exc)))
    return BatchSummary(tuple(results))
