"""Turn timestamped OCR observations into filtered, merged SRT entries."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from rapidfuzz.fuzz import ratio

if TYPE_CHECKING:
    from .ocr_engine import RecognizedText


@dataclass(frozen=True)
class TextObservation:
    text: str
    confidence: float
    # Normalized x_min, y_min, x_max, y_max. Needed for static watermark tracks.
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FrameObservation:
    timestamp_seconds: float
    texts: tuple[TextObservation, ...]


def frame_from_ocr(
    timestamp_seconds: float, width: int, height: int, lines: Sequence[RecognizedText]
) -> FrameObservation:
    """Convert pixel-space OCR boxes to observations for temporal merging."""
    if width <= 0 or height <= 0:
        raise ValueError("Kích thước khung hình không hợp lệ")
    texts = []
    for line in lines:
        x1, y1, x2, y2 = line.box.bounds
        texts.append(TextObservation(
            line.text, line.score,
            (max(0.0, x1 / width), max(0.0, y1 / height),
             min(1.0, x2 / width), min(1.0, y2 / height)),
        ))
    return FrameObservation(timestamp_seconds, tuple(texts))


@dataclass(frozen=True)
class SubtitleEntry:
    id: int
    start_ms: int
    end_ms: int
    text: str

    def __post_init__(self) -> None:
        if self.id < 1 or self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("ID hoặc thời gian phụ đề không hợp lệ")
        if not self.text.strip():
            raise ValueError("Nội dung phụ đề trống")


@dataclass(frozen=True)
class BuilderConfig:
    similarity_threshold: float = 85.0
    watermark_seconds: float = 15.0
    minimum_duration_seconds: float = 0.15
    bridge_gap_seconds: float = 0.12
    minimum_samples: int = 2
    position_tolerance: float = 0.025
    watermark_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.similarity_threshold <= 100:
            raise ValueError("similarity_threshold phải nằm trong [0, 100]")
        if self.watermark_seconds <= 0 or self.minimum_duration_seconds < 0:
            raise ValueError("Ngưỡng thời lượng không hợp lệ")
        if self.bridge_gap_seconds < 0 or self.minimum_samples < 1:
            raise ValueError("Ngưỡng lọc nhiễu không hợp lệ")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _looks_like_url(text: str) -> bool:
    value = text.strip().casefold()
    return (value.startswith(("http", "www."))
            or bool(re.search(r"\b[a-z0-9-]+\.(?:com|cc|org|net|top|app)\b", value)))


def _same_position(a: TextObservation, b: TextObservation, tolerance: float) -> bool:
    return a.box is not None and b.box is not None and all(
        abs(x - y) <= tolerance for x, y in zip(a.box, b.box)
    )


def _sample_interval(frames: Sequence[FrameObservation], fallback: float) -> float:
    deltas = [b.timestamp_seconds - a.timestamp_seconds for a, b in zip(frames, frames[1:])]
    return statistics.median(deltas) if deltas else fallback


def _watermark_indexes(
    frames: Sequence[FrameObservation], interval: float, config: BuilderConfig
) -> set[tuple[int, int]]:
    """Find identical text at the same position in consecutive sampled frames."""
    removed: set[tuple[int, int]] = set()
    tracks: list[list[tuple[int, int]]] = []

    def finish(track: list[tuple[int, int]]) -> None:
        if not track:
            return
        first, last = track[0][0], track[-1][0]
        duration = frames[last].timestamp_seconds - frames[first].timestamp_seconds + interval
        if duration > config.watermark_seconds:
            removed.update(track)

    for frame_index, frame in enumerate(frames):
        used: set[int] = set()
        continued: list[list[tuple[int, int]]] = []
        for track in tracks:
            last_frame, last_text = track[-1]
            previous = frames[last_frame].texts[last_text]
            match = next(
                (text_index for text_index, current in enumerate(frame.texts)
                 if text_index not in used
                 and frame_index == last_frame + 1
                 and frame.timestamp_seconds - frames[last_frame].timestamp_seconds <= interval * 1.6
                 and _normalized(previous.text) == _normalized(current.text)
                 and _same_position(previous, current, config.position_tolerance)),
                None,
            )
            if match is None:
                finish(track)
            else:
                used.add(match)
                continued.append(track + [(frame_index, match)])
        for text_index, item in enumerate(frame.texts):
            if text_index not in used and item.text.strip() and item.box is not None:
                continued.append([(frame_index, text_index)])
        tracks = continued
    for track in tracks:
        finish(track)
    return removed


def _frame_text(frame: FrameObservation, frame_index: int, removed: set[tuple[int, int]],
                config: BuilderConfig) -> tuple[str, float]:
    present = [item for index, item in enumerate(frame.texts)
               if (frame_index, index) not in removed and item.text.strip()
               and not _looks_like_url(item.text)
               and not any(term.casefold() in item.text.casefold()
                           for term in config.watermark_terms if term.strip())]
    if not present:
        return "", 0.0
    # Keep detector reading order. OCR engines normally return top-to-bottom lines.
    text = "\n".join(item.text.strip() for item in present)
    confidence = sum(max(0.0, min(1.0, item.confidence)) for item in present) / len(present)
    return text, confidence


def build_subtitles(
    frames: Iterable[FrameObservation], *, config: BuilderConfig | None = None,
    sample_fps: float = 3.0,
) -> list[SubtitleEntry]:
    """Merge OCR frames and remove persistent text, short flicker and tiny gaps."""
    cfg = config or BuilderConfig()
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps phải là số dương hữu hạn")
    ordered = sorted(frames, key=lambda frame: frame.timestamp_seconds)
    if not ordered:
        return []
    if any(not math.isfinite(frame.timestamp_seconds) or frame.timestamp_seconds < 0 for frame in ordered):
        raise ValueError("Mốc thời gian OCR không hợp lệ")
    if any(a.timestamp_seconds == b.timestamp_seconds for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Mốc thời gian OCR bị trùng")
    interval = _sample_interval(ordered, 1 / sample_fps)
    removed = _watermark_indexes(ordered, interval, cfg)
    entries: list[SubtitleEntry] = []
    first: float | None = None
    last = 0.0
    best_text = ""
    best_confidence = -1.0
    count = 0

    def flush() -> None:
        nonlocal first, last, best_text, best_confidence, count
        if first is None:
            return
        end = last + interval
        if count >= cfg.minimum_samples and end - first >= cfg.minimum_duration_seconds:
            start_ms = round(first * 1000)
            end_ms = max(start_ms + 1, round(end * 1000))
            entries.append(SubtitleEntry(len(entries) + 1, start_ms, end_ms, best_text))
        first, last, best_text, best_confidence, count = None, 0.0, "", -1.0, 0

    for frame_index, frame in enumerate(ordered):
        text, confidence = _frame_text(frame, frame_index, removed, cfg)
        if not text:
            continue
        if first is not None:
            gap = frame.timestamp_seconds - (last + interval)
            same_text = ratio(_normalized(text), _normalized(best_text)) >= cfg.similarity_threshold
            if not same_text or gap > cfg.bridge_gap_seconds + 1e-9:
                flush()
        if first is None:
            first = frame.timestamp_seconds
        last = frame.timestamp_seconds
        count += 1
        if confidence > best_confidence:
            best_confidence, best_text = confidence, text
    flush()
    return entries


def _timecode(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_srt(entries: Iterable[SubtitleEntry]) -> str:
    blocks = [f"{entry.id}\n{_timecode(entry.start_ms)} --> {_timecode(entry.end_ms)}\n{entry.text.strip()}"
              for entry in entries]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(entries: Iterable[SubtitleEntry], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_srt(entries), encoding="utf-8-sig")
    return destination
