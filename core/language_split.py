"""Route OCR lines to language-specific subtitle tracks.

The current multilingual OCR model does not report a language per line. Script
is reliable for Chinese/Japanese versus Latin; Vietnamese diacritics distinguish
Vietnamese from English. Plain ASCII Vietnamese remains ambiguous in auto mode.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from .srt_builder import FrameObservation, SubtitleEntry

LANGUAGE_CODES = ("zh", "vi", "ja", "en", "und")
VIETNAMESE_CHARS = frozenset(
    "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợ"
    "ùúủũụừứửữựỳýỷỹỵĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
)


def language_of(text: str, *, latin_fallback: str = "en") -> str | None:
    """Classify a single OCR line; return None for numbers and punctuation."""
    text = re.sub(r"\{\\[^}]+\}", "", text)  # ASS position tags are not content.
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    if any("\u3400" <= char <= "\u9fff" for char in text):
        return "zh"
    if any(char in VIETNAMESE_CHARS for char in text):
        return "vi"
    if any(char.isalpha() for char in text):
        return latin_fallback
    return None


def split_observations(
    frames: Iterable[FrameObservation], *, selected_language: str = "auto",
) -> tuple[dict[str, list[FrameObservation]], str]:
    """Preserve every timestamp in each language track for temporal merging."""
    ordered = list(frames)
    latin_fallback = "vi" if selected_language == "vi" else "en"
    votes: Counter[str] = Counter()
    routes: list[list[str | None]] = []
    for frame in ordered:
        languages = [language_of(text.text, latin_fallback=latin_fallback) for text in frame.texts]
        routes.append(languages)
        first = next((language for language in languages if language is not None), None)
        if first:
            votes[first] += 1
    primary = votes.most_common(1)[0][0] if votes else {
        "ch": "zh", "ja": "ja", "vi": "vi", "en": "en",
    }.get(selected_language, "und")
    detected = {language for languages in routes for language in languages if language is not None}
    if not detected or any(None in languages for languages in routes):
        detected.add(primary)
    tracks: dict[str, list[FrameObservation]] = {language: [] for language in detected or {primary}}
    for frame, languages in zip(ordered, routes):
        grouped: dict[str, list] = {}
        for text, language in zip(frame.texts, languages):
            grouped.setdefault(language or primary, []).append(text)
        for language in tracks:
            tracks[language].append(FrameObservation(
                frame.timestamp_seconds, tuple(grouped.get(language, ())),
            ))
    return tracks, primary


def split_entries(
    entries: Iterable[SubtitleEntry], *, primary: str, selected_language: str = "auto",
) -> dict[str, list[SubtitleEntry]]:
    """Route editorial captions and messages by their own language."""
    latin_fallback = "vi" if selected_language == "vi" else "en"
    tracks: dict[str, list[SubtitleEntry]] = {}
    for entry in entries:
        language = language_of(entry.text, latin_fallback=latin_fallback) or primary
        tracks.setdefault(language, []).append(entry)
    return tracks
