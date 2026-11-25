"""Optional, conservative OCR text correction via an OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

import requests
from rapidfuzz.fuzz import ratio

from .srt_builder import SubtitleEntry

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You correct OCR errors in subtitles. Only fix obvious character errors, broken "
    "spacing, and punctuation using context. Never translate, merge, split, delete, "
    "or reorder subtitles. Never change an ID or timestamp. Return only one JSON "
    "object whose keys are the exact input IDs as strings and whose values are the "
    "corrected text strings. Include every ID exactly once. If uncertain, preserve "
    "the original text."
)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    batch_size: int = 30
    timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 100 or self.timeout_seconds <= 0:
            raise ValueError("Cỡ lô hoặc thời gian chờ LLM không hợp lệ")
        if self.enabled and (not self.base_url or not self.model):
            raise ValueError("Khi bật LLM phải có base_url và model")


def _content_to_map(content: object, expected_ids: set[str]) -> dict[str, str]:
    if not isinstance(content, str):
        raise ValueError("LLM không trả nội dung văn bản")
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    data = json.loads(stripped)
    if not isinstance(data, dict) or set(data) != expected_ids:
        raise ValueError("LLM đã thay đổi hoặc bỏ sót ID")
    if any(not isinstance(value, str) or not value.strip() for value in data.values()):
        raise ValueError("LLM trả nội dung phụ đề trống hoặc sai kiểu")
    return data


def _plausible_correction(original: str, corrected: str) -> bool:
    """Reject obvious translations or rewrites despite an otherwise valid JSON map."""
    source = " ".join(original.casefold().split())
    target = " ".join(corrected.casefold().split())
    if len(source) >= 4 and ratio(source, target) < 35:
        return False
    has_cjk = lambda value: any(
        "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" for char in value
    )
    if has_cjk(source) and not has_cjk(target):
        return False
    return True


def refine_subtitles(
    entries: Sequence[SubtitleEntry], config: LLMConfig | None = None,
    *, session: requests.Session | None = None,
) -> list[SubtitleEntry]:
    """Correct only text; preserve IDs/times and keep originals on API errors."""
    cfg = config or LLMConfig()
    originals = list(entries)
    if not cfg.enabled or not originals:
        return originals
    assert cfg.base_url is not None and cfg.model is not None
    endpoint = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    client = session or requests.Session()
    refined: list[SubtitleEntry] = []
    for start in range(0, len(originals), cfg.batch_size):
        batch = originals[start:start + cfg.batch_size]
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    [{"id": item.id, "start_ms": item.start_ms, "end_ms": item.end_ms,
                      "text": item.text} for item in batch], ensure_ascii=False)},
            ],
            "stream": False,
        }
        try:
            response = client.post(endpoint, headers=headers, json=payload, timeout=cfg.timeout_seconds)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            texts = _content_to_map(content, {str(item.id) for item in batch})
            if any(not _plausible_correction(item.text, texts[str(item.id)]) for item in batch):
                raise ValueError("LLM đã viết lại nội dung quá khác bản OCR")
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            LOGGER.warning("Bỏ qua sửa OCR bằng LLM cho lô %s–%s: %s", batch[0].id, batch[-1].id, exc)
            refined.extend(batch)
            continue
        refined.extend(SubtitleEntry(item.id, item.start_ms, item.end_ms, texts[str(item.id)].strip())
                       for item in batch)
    return refined
