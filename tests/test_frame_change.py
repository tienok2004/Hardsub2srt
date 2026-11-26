import random
import unittest

from PIL import Image, ImageDraw

from core.frame_change import FrameChangeDetector
from core.video_sampler import SampledFrame


def _frame(text: str, *, noise: int = 0, seed: int = 0, size=(1920, 324)) -> SampledFrame:
    image = Image.new("RGB", size, (40, 60, 80))
    if noise:
        rng = random.Random(seed)
        pixels = image.load()
        for _ in range(4000):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            shade = 40 + rng.randint(-noise, noise)
            pixels[x, y] = (shade, 60, 80)
    ImageDraw.Draw(image).text((700, 150), text, fill="white", font_size=40)
    return SampledFrame(0.0, size[0], size[1], image.tobytes())


class FrameChangeTests(unittest.TestCase):
    def test_first_frame_always_runs_ocr(self) -> None:
        self.assertTrue(FrameChangeDetector().changed(_frame("Xin chào")))

    def test_same_text_with_grain_is_skipped(self) -> None:
        detector = FrameChangeDetector()
        detector.changed(_frame("Xin chào", noise=6, seed=1))
        self.assertFalse(detector.changed(_frame("Xin chào", noise=6, seed=2)))

    def test_punctuation_change_is_detected(self) -> None:
        detector = FrameChangeDetector()
        detector.changed(_frame("Được rồi."))
        self.assertTrue(detector.changed(_frame("Được rồi!")))

    def test_subtitle_disappearing_is_detected(self) -> None:
        detector = FrameChangeDetector()
        detector.changed(_frame("Tạm biệt"))
        self.assertTrue(detector.changed(_frame("")))

    def test_slow_fade_accumulates_against_last_ocr_frame(self) -> None:
        detector = FrameChangeDetector()
        detector.changed(_frame(""))
        results = []
        for level in range(1, 11):
            image = Image.new("RGB", (1920, 324), (40, 60, 80))
            shade = 40 + level * 20
            ImageDraw.Draw(image).text((700, 150), "Fade", fill=(shade, shade, shade), font_size=40)
            results.append(detector.changed(SampledFrame(0.0, 1920, 324, image.tobytes())))
        self.assertFalse(results[0])  # one faint step is below the threshold
        self.assertTrue(any(results))


if __name__ == "__main__":
    unittest.main()
