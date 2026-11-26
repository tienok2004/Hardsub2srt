"""Editorial captions ("Hà Nội, 1975", "Ba năm sau") read outside the dialogue area.

Text outside the dialogue band is a mix of what we want (captions added in
editing) and what we do not (site logos, animated watermarks, ad banners, shop
signs, brand names on clothing). Each piece of text is followed over time as a
*track*; a track becomes a caption only if it behaves like an overlay:

* it stays put (overlay text is pixel-stable; signs and clothing move with the
  camera or the actor),
* it is level (scene text is often tilted or in perspective),
* it lasts long enough to be read but not so long that it is a logo,
* it is read with high confidence at a size close to the dialogue font,
* it is not a URL, an ad phrase, or a user-listed watermark term,
* the same text does not come back again and again (animated / rotating logos).
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rapidfuzz.fuzz import ratio

from .srt_builder import SubtitleEntry, _looks_like_url, _normalized

TOP_TAG = r"{\an8}"  # SRT/ASS position tag: show at the top of the screen

DEFAULT_AD_TERMS = (
    "www", "http", ".com", ".net", ".vn", ".tv", ".cc", ".top", ".app",
    "subscribe", "đăng ký kênh", "theo dõi kênh", "tải app", "xem phim tại", "xem full tại",
    "phim mới tại", "hotline", "zalo", "tiktok", "facebook", "youtube", "fanpage", "telegram",
    "关注", "订阅", "下载", "官网", "抖音", "微博", "公众号", "快手",
)


@dataclass(frozen=True)
class CaptionText:
    text: str
    confidence: float
    box: tuple[float, float, float, float]  # normalized to the full frame
    angle_degrees: float = 0.0


@dataclass(frozen=True)
class CaptionFrame:
    timestamp_seconds: float
    texts: tuple[CaptionText, ...]


@dataclass(frozen=True)
class CaptionConfig:
    min_seconds: float = 1.0
    max_seconds: float = 15.0  # longer than this at one spot: logo/watermark
    max_jitter: float = 0.012  # largest drift of the box centre, as a share of the frame
    max_angle_degrees: float = 3.0
    min_confidence: float = 0.80
    max_repeats: int = 3  # same text appearing this many times: rotating logo/ad
    min_height: float = 0.02  # text height as a share of frame height
    max_height: float = 0.12
    dialogue_height_ratio: tuple[float, float] = (0.45, 2.2)
    ad_terms: tuple[str, ...] = DEFAULT_AD_TERMS
    zones: tuple[tuple[float, float, float, float], ...] = ((0.0, 0.0, 1.0, 0.35),)
    top_tag: bool = True

    def __post_init__(self) -> None:
        if self.min_seconds < 0 or self.max_seconds <= self.min_seconds:
            raise ValueError("Ngưỡng thời lượng chú thích không hợp lệ")
        if self.max_repeats < 2:
            raise ValueError("max_repeats phải từ 2 trở lên")
        for zone in self.zones:
            if len(zone) != 4 or not (0 <= zone[0] < zone[2] <= 1 and 0 <= zone[1] < zone[3] <= 1):
                raise ValueError(f"Vùng chú thích không hợp lệ: {zone}")


def caption_config_from(settings: dict | None, watermark_seconds: float = 15.0) -> CaptionConfig:
    """Build the filter settings from the ``captions:`` section of config.yaml."""
    settings = settings or {}
    defaults = CaptionConfig()
    zones = settings.get("zones")
    return CaptionConfig(
        min_seconds=float(settings.get("min_seconds", defaults.min_seconds)),
        max_seconds=float(settings.get("max_seconds", watermark_seconds)),
        max_jitter=float(settings.get("max_jitter", defaults.max_jitter)),
        max_angle_degrees=float(settings.get("max_angle_degrees", defaults.max_angle_degrees)),
        min_confidence=float(settings.get("min_confidence", defaults.min_confidence)),
        max_repeats=int(settings.get("max_repeats", defaults.max_repeats)),
        ad_terms=tuple(settings.get("ad_terms", defaults.ad_terms)),
        zones=tuple(tuple(float(v) for v in zone) for zone in zones) if zones else defaults.zones,
        top_tag=bool(settings.get("top_tag", defaults.top_tag)),
    )


@dataclass
class _Track:
    items: list[tuple[int, float, CaptionText]] = field(default_factory=list)  # (frame, time, text)

    @property
    def last_frame(self) -> int:
        return self.items[-1][0]

    def best(self) -> CaptionText:
        return max((item[2] for item in self.items), key=lambda text: text.confidence)


def angle_of(points: Sequence[tuple[float, float]]) -> float:
    """Tilt of a detected text quad's top edge, in degrees."""
    if len(points) < 2:
        return 0.0
    (x1, y1), (x2, y2) = points[0], points[1]
    if x2 == x1:
        return 90.0
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _letters(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _is_ad(text: str, terms: Iterable[str]) -> bool:
    value = _normalized(text)
    return _looks_like_url(value) or any(term.casefold() in value for term in terms if term.strip())


def _tracks(frames: Sequence[CaptionFrame]) -> list[_Track]:
    finished: list[_Track] = []
    active: list[_Track] = []
    for index, frame in enumerate(frames):
        still: list[_Track] = []
        used: set[int] = set()
        for track in active:
            last = track.items[-1][2]
            match = next((i for i, text in enumerate(frame.texts)
                          if i not in used and _iou(last.box, text.box) >= 0.4
                          and ratio(_normalized(last.text), _normalized(text.text)) >= 80), None)
            if match is not None:
                used.add(match)
                track.items.append((index, frame.timestamp_seconds, frame.texts[match]))
                still.append(track)
            elif index - track.last_frame <= 1:
                still.append(track)  # allow one missed sample (OCR flicker)
            else:
                finished.append(track)
        for i, text in enumerate(frame.texts):
            if i not in used and text.text.strip():
                still.append(_Track([(index, frame.timestamp_seconds, text)]))
        active = still
    return finished + active


def _rejection(track: _Track, interval: float, config: CaptionConfig, dialogue_height: float | None,
               watermark_terms: Sequence[str]) -> str | None:
    """Why this track is not an editorial caption, or None when it is one."""
    texts = [item[2] for item in track.items]
    best = track.best()
    duration = track.items[-1][1] - track.items[0][1] + interval
    if len(texts) < 2 or duration < config.min_seconds:
        return "quá ngắn"
    if duration > config.max_seconds:
        return "đứng yên quá lâu (logo)"
    if _letters(best.text) < 2:
        return "quá ít chữ"
    if _is_ad(best.text, config.ad_terms) or any(
            term.casefold() in best.text.casefold() for term in watermark_terms if term.strip()):
        return "quảng cáo / watermark"
    if statistics.mean(text.confidence for text in texts) < config.min_confidence:
        return "độ tin cậy thấp"
    if abs(statistics.median(text.angle_degrees for text in texts)) > config.max_angle_degrees:
        return "chữ nghiêng (biển hiệu, áo quần)"
    centers = [((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2) for t in texts]
    mid_x = statistics.median(c[0] for c in centers)
    mid_y = statistics.median(c[1] for c in centers)
    if max(max(abs(x - mid_x), abs(y - mid_y)) for x, y in centers) > config.max_jitter:
        return "di chuyển theo cảnh"
    height = statistics.median(t.box[3] - t.box[1] for t in texts)
    if not config.min_height <= height <= config.max_height:
        return "cỡ chữ bất thường"
    if dialogue_height:
        low, high = config.dialogue_height_ratio
        if not low <= height / dialogue_height <= high:
            return "cỡ chữ khác xa phụ đề"
    return None


def build_captions(
    frames: Iterable[CaptionFrame], *, interval: float, config: CaptionConfig | None = None,
    dialogue_height: float | None = None, watermark_terms: Sequence[str] = (),
) -> tuple[list[SubtitleEntry], dict[str, int]]:
    """Return caption entries (ids from 1) and how many tracks each filter dropped."""
    cfg = config or CaptionConfig()
    ordered = sorted(frames, key=lambda frame: frame.timestamp_seconds)
    tracks = _tracks(ordered)
    rejected: dict[str, int] = {}
    accepted: list[_Track] = []
    for track in tracks:
        reason = _rejection(track, interval, cfg, dialogue_height, watermark_terms)
        if reason is None:
            accepted.append(track)
        elif reason != "quá ngắn":
            rejected[reason] = rejected.get(reason, 0) + 1
    # Rotating logos and recurring ad banners pass every per-track test, but the
    # same text keeps coming back. Count all readable appearances, short ones too.
    readable = [track for track in tracks if _letters(track.best().text) >= 2]
    kept: list[_Track] = []
    for track in accepted:
        text = _normalized(track.best().text)
        repeats = sum(ratio(text, _normalized(other.best().text)) >= 90 for other in readable)
        if repeats >= cfg.max_repeats:
            rejected["lặp lại nhiều lần (logo động)"] = rejected.get("lặp lại nhiều lần (logo động)", 0) + 1
        else:
            kept.append(track)
    entries: list[SubtitleEntry] = []
    for track in sorted(kept, key=lambda item: item.items[0][1]):
        best = track.best()
        start = track.items[0][1]
        end = track.items[-1][1] + interval
        center_y = (best.box[1] + best.box[3]) / 2
        text = best.text.strip()
        if cfg.top_tag and center_y < 0.5:
            text = TOP_TAG + text
        start_ms = round(start * 1000)
        entries.append(SubtitleEntry(len(entries) + 1, start_ms, max(start_ms + 1, round(end * 1000)), text))
    return entries, rejected


def merge_entries(dialogue: Sequence[SubtitleEntry], captions: Sequence[SubtitleEntry]) -> list[SubtitleEntry]:
    """One SRT with both tracks, ordered by start time and renumbered.

    Entries may overlap in time; SRT players show overlapping entries together,
    and the ``{\\an8}`` tag keeps captions at the top, away from the dialogue.
    """
    ordered = sorted([*dialogue, *captions], key=lambda entry: (entry.start_ms, entry.end_ms))
    return [SubtitleEntry(index, entry.start_ms, entry.end_ms, entry.text)
            for index, entry in enumerate(ordered, 1)]
