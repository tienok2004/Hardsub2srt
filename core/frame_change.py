"""Detect when the subtitle area actually changed, so OCR can skip repeats.

A subtitle line usually stays on screen for 2–4 seconds, i.e. 6–12 samples at
3 FPS. Re-running OCR on identical pixels is the main waste in the pipeline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .video_sampler import SampledFrame

THUMBNAIL_WIDTH = 640
PIXEL_DELTA = 40
MIN_CHANGED_PIXELS = 8


def _thumbnail(frame: SampledFrame) -> np.ndarray:
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb24).convert("L")
    width = min(THUMBNAIL_WIDTH, frame.width)
    height = max(1, round(frame.height * width / frame.width))
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.int16)


class FrameChangeDetector:
    """Compare each frame with the last frame that was sent to OCR.

    Comparing against the last OCR'd frame (not the previous sample) means slow
    fades still accumulate into a detected change. Measured on 1080p test clips:
    unchanged frames with grain/compression noise differ by at most ~6 grey
    levels, while even a "." → "!" swap changes 12+ pixels by more than 40.
    """

    def __init__(self, *, pixel_delta: int = PIXEL_DELTA, min_changed_pixels: int = MIN_CHANGED_PIXELS) -> None:
        self.pixel_delta = pixel_delta
        self.min_changed_pixels = min_changed_pixels
        self._reference: np.ndarray | None = None

    def changed(self, frame: SampledFrame) -> bool:
        current = _thumbnail(frame)
        reference = self._reference
        if reference is None or reference.shape != current.shape:
            self._reference = current
            return True
        changed_pixels = int(np.count_nonzero(np.abs(current - reference) > self.pixel_delta))
        if changed_pixels >= self.min_changed_pixels:
            self._reference = current
            return True
        return False
