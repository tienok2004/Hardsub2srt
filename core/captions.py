"""Editorial captions and screen messages read outside the dialogue area.

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
    vertical: bool = False


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


@dataclass(frozen=True)
class ScreenTextConfig:
    """Readable messages on a phone or another display in the middle of a scene."""

    enabled: bool = True
    zones: tuple[tuple[float, float, float, float], ...] = ((0.12, 0.28, 0.88, 0.78),)
    min_seconds: float = 0.65
    max_seconds: float = 15.0
    min_chars: int = 5
    min_confidence: float = 0.80
    max_jitter: float = 0.09
    max_angle_degrees: float = 30.0
    min_height: float = 0.015
    max_height: float = 0.12
    max_repeats: int = 3

    def __post_init__(self) -> None:
        if self.min_seconds < 0 or self.max_seconds <= self.min_seconds or self.min_chars < 2:
            raise ValueError("Ngưỡng chữ trên màn hình không hợp lệ")
        if self.max_repeats < 2:
            raise ValueError("max_repeats phải từ 2 trở lên")
        for zone in self.zones:
            if len(zone) != 4 or not (0 <= zone[0] < zone[2] <= 1 and 0 <= zone[1] < zone[3] <= 1):
                raise ValueError(f"Vùng chữ trên màn hình không hợp lệ: {zone}")


def screen_text_config_from(settings: dict | None) -> ScreenTextConfig:
    settings = settings or {}
    defaults = ScreenTextConfig()
    zones = settings.get("zones")
    return ScreenTextConfig(
        enabled=bool(settings.get("enabled", defaults.enabled)),
        zones=tuple(tuple(float(v) for v in zone) for zone in zones) if zones is not None else defaults.zones,
        min_seconds=float(settings.get("min_seconds", defaults.min_seconds)),
        max_seconds=float(settings.get("max_seconds", defaults.max_seconds)),
        min_chars=int(settings.get("min_chars", defaults.min_chars)),
        min_confidence=float(settings.get("min_confidence", defaults.min_confidence)),
        max_jitter=float(settings.get("max_jitter", defaults.max_jitter)),
        max_angle_degrees=float(settings.get("max_angle_degrees", defaults.max_angle_degrees)),
        max_repeats=int(settings.get("max_repeats", defaults.max_repeats)),
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


def is_corner_overlay(box: tuple[float, float, float, float]) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (center_x < 0.35 or center_x > 0.75) and (center_y < 0.15 or center_y > 0.85)


def _is_ad(text: str, terms: Iterable[str]) -> bool:
    value = _normalized(text)
    return _looks_like_url(value) or any(term.casefold() in value for term in terms if term.strip())


def _beside_site_address(track: _Track, frames: Sequence[CaptionFrame]) -> bool:
    """A short name printed beside a domain is usually a site badge."""
    if _letters(track.best().text) > 12:
        return False
    matches = 0
    for frame_index, _, text in track.items:
        if any(_looks_like_url(other.text)
               and abs((text.box[1] + text.box[3] - other.box[1] - other.box[3]) / 2) <= 0.07
               and text.box[0] <= other.box[2] + 0.06
               and other.box[0] <= text.box[2] + 0.06
               for other in frames[frame_index].texts):
            matches += 1
    return matches >= min(2, len(track.items))


def _tracks(frames: Sequence[CaptionFrame], *, min_iou: float = 0.4,
            min_similarity: float = 80) -> list[_Track]:
    finished: list[_Track] = []
    active: list[_Track] = []
    for index, frame in enumerate(frames):
        still: list[_Track] = []
        used: set[int] = set()
        for track in active:
            last = track.items[-1][2]
            match = next((i for i, text in enumerate(frame.texts)
                          if i not in used and _iou(last.box, text.box) >= min_iou
                          and ratio(_normalized(last.text), _normalized(text.text)) >= min_similarity), None)
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
    if best.vertical and sum("\u3400" <= char <= "\u9fff" for char in best.text) < 2:
        return "chữ dọc không phải chú thích"
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
    if not best.vertical and not config.min_height <= height <= config.max_height:
        return "cỡ chữ bất thường"
    if dialogue_height and not best.vertical:
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
        if reason is None and _beside_site_address(track, ordered):
            reason = "logo cạnh địa chỉ web"
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
            best = track.best()
            if not best.vertical and is_corner_overlay(best.box) and _letters(best.text) <= 6:
                rejected["logo ở góc khung hình"] = rejected.get("logo ở góc khung hình", 0) + 1
            else:
                kept.append(track)
    groups: list[list[_Track]] = []
    for track in sorted(kept, key=lambda item: item.items[0][1]):
        if track.best().vertical:
            match = next((group for group in groups if group[0].best().vertical and any(
                _vertical_neighbors(track, other, interval, cfg.min_seconds) for other in group)), None)
            if match is not None:
                match.append(track)
                continue
        groups.append([track])
    entries: list[SubtitleEntry] = []
    for group in sorted(groups, key=lambda items: min(track.items[0][1] for track in items)):
        ordered = sorted(group, key=lambda track: track.best().box[0])
        if group[0].best().vertical:
            columns: list[_Track] = []
            for track in ordered:
                duplicate = next((index for index, other in enumerate(columns)
                                  if _iou(track.best().box, other.best().box) >= 0.6), None)
                if duplicate is None:
                    columns.append(track)
                elif track.best().confidence > columns[duplicate].best().confidence:
                    columns[duplicate] = track
            ordered = sorted(columns, key=lambda track: track.best().box[0])
        start = min(track.items[0][1] for track in group)
        end = max(track.items[-1][1] + interval for track in group)
        center_y = min((track.best().box[1] + track.best().box[3]) / 2 for track in group)
        text = "：".join(track.best().text.strip() for track in ordered)
        if not (text.startswith("(") and text.endswith(")")):
            text = f"({text})"
        if cfg.top_tag and center_y < 0.5:
            text = TOP_TAG + text
        start_ms = round(start * 1000)
        entries.append(SubtitleEntry(len(entries) + 1, start_ms, max(start_ms + 1, round(end * 1000)), text))
    return entries, rejected


def _vertical_neighbors(a: _Track, b: _Track, interval: float, min_seconds: float) -> bool:
    first, second = a.best().box, b.best().box
    horizontal_gap = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    vertical_overlap = min(first[3], second[3]) - max(first[1], second[1])
    temporal_overlap = min(a.items[-1][1], b.items[-1][1]) + interval - max(a.items[0][1], b.items[0][1])
    return horizontal_gap <= 0.06 and vertical_overlap >= 0.04 and temporal_overlap >= min_seconds / 2


def build_screen_text(
    frames: Iterable[CaptionFrame], *, interval: float, config: ScreenTextConfig | None = None,
    ad_terms: Sequence[str] = DEFAULT_AD_TERMS, watermark_terms: Sequence[str] = (),
) -> tuple[list[SubtitleEntry], dict[str, int]]:
    """Keep transient, readable screen messages and wrap them in parentheses."""
    cfg = config or ScreenTextConfig()
    if not cfg.enabled:
        return [], {}
    ordered = sorted(frames, key=lambda frame: frame.timestamp_seconds)
    tracks = _tracks(ordered,
                     min_iou=0.2, min_similarity=75)
    rejected: dict[str, int] = {}
    accepted: list[_Track] = []
    for track in tracks:
        texts = [item[2] for item in track.items]
        best = track.best()
        duration = track.items[-1][1] - track.items[0][1] + interval
        reason = None
        if len(texts) < 2 or duration < cfg.min_seconds:
            reason = "quá ngắn"
        elif duration > cfg.max_seconds:
            reason = "đứng yên quá lâu (logo)"
        elif _letters(best.text) < cfg.min_chars or not any(ch.isalpha() for ch in best.text):
            reason = "không giống nội dung tin nhắn"
        elif _is_ad(best.text, ad_terms) or any(
                term.casefold() in best.text.casefold() for term in watermark_terms if term.strip()):
            reason = "quảng cáo / watermark"
        elif _beside_site_address(track, ordered):
            reason = "logo cạnh địa chỉ web"
        elif statistics.mean(text.confidence for text in texts) < cfg.min_confidence:
            reason = "độ tin cậy thấp"
        elif abs(statistics.median(text.angle_degrees for text in texts)) > cfg.max_angle_degrees:
            reason = "chữ nghiêng quá mức"
        else:
            centers = [((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2) for t in texts]
            mid_x = statistics.median(c[0] for c in centers)
            mid_y = statistics.median(c[1] for c in centers)
            height = statistics.median(t.box[3] - t.box[1] for t in texts)
            if max(max(abs(x - mid_x), abs(y - mid_y)) for x, y in centers) > cfg.max_jitter:
                reason = "di chuyển quá nhiều"
            elif not cfg.min_height <= height <= cfg.max_height:
                reason = "cỡ chữ bất thường"
        if reason is None:
            accepted.append(track)
        elif reason != "quá ngắn":
            rejected[reason] = rejected.get(reason, 0) + 1
    readable = [track for track in tracks if _letters(track.best().text) >= cfg.min_chars]
    entries: list[SubtitleEntry] = []
    for track in sorted(accepted, key=lambda item: item.items[0][1]):
        best = track.best()
        repeated = sum(ratio(_normalized(best.text), _normalized(other.best().text)) >= 90
                       for other in readable)
        if repeated >= cfg.max_repeats:
            rejected["lặp lại nhiều lần (logo động)"] = rejected.get("lặp lại nhiều lần (logo động)", 0) + 1
            continue
        start_ms = round(track.items[0][1] * 1000)
        end_ms = max(start_ms + 1, round((track.items[-1][1] + interval) * 1000))
        content = best.text.strip()
        if not (content.startswith("(") and content.endswith(")")):
            content = f"({content})"
        entries.append(SubtitleEntry(len(entries) + 1, start_ms, end_ms, content))
    return entries, rejected


def merge_entries(dialogue: Sequence[SubtitleEntry], captions: Sequence[SubtitleEntry]) -> list[SubtitleEntry]:
    """One SRT with both tracks, ordered by start time and renumbered.

    Entries may overlap in time; SRT players show overlapping entries together.
    Editorial captions near the top carry an ``{\\an8}`` position tag.
    """
    ordered = sorted([*dialogue, *captions], key=lambda entry: (entry.start_ms, entry.end_ms))
    return [SubtitleEntry(index, entry.start_ms, entry.end_ms, entry.text)
            for index, entry in enumerate(ordered, 1)]
