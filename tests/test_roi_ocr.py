"""Stage 2 checks: multilingual OCR and ROI behavior on real video frames."""

import tempfile
import subprocess
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.ocr_engine import OCREngine, TextBox
from core.roi_detector import FULL_FRAME, ROI, _best_roi, detect_roi


class _SparseDetector:
    def __init__(self, hits: set[int]) -> None:
        self.hits = hits
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls not in self.hits:
            return []
        return [TextBox(((40, 65), (130, 65), (130, 84), (40, 84)), 0.9)]


class StageTwoTests(unittest.TestCase):
    def test_static_corner_logo_does_not_override_subtitle_zone(self) -> None:
        observations = []
        for index in range(30):
            observations.append((index, index * 10.0,
                                 (0.03, 0.02, 0.34 + (index % 2) * 0.03, 0.08), False))
            if index % 3 != 0:
                observations.append((index, index * 10.0,
                                     (0.15 + index % 5 * 0.04, 0.78, 0.75, 0.84), False))
        result = _best_roi(observations, minimum_support_frames=2, padding_ratio=0.03)
        self.assertIsNotNone(result)
        self.assertGreater(result[0].y_min, 0.6)

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.ffmpeg = next((str(path) for path in (root / '.tools' / 'ffmpeg').rglob('ffmpeg.exe')), None)
        cls.ffprobe = next((str(path) for path in (root / '.tools' / 'ffmpeg').rglob('ffprobe.exe')), None)
        if not cls.ffmpeg or not cls.ffprobe:
            raise unittest.SkipTest('Local FFmpeg unavailable')
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / 'clip.mp4'
        subprocess.run([cls.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'testsrc=size=160x90:rate=30:duration=2', '-c:v', 'mpeg4',
                        '-y', str(cls.video)], capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_multilingual_ocr_recognizes_vietnamese_diacritics(self) -> None:
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        if not font_path.is_file():
            self.skipTest("Arial font unavailable")
        image = Image.new("RGB", (960, 200), "black")
        ImageDraw.Draw(image).text((50, 55), "Xin chào Việt Nam", fill="white",
                                   font=ImageFont.truetype(str(font_path), 58))
        lines = OCREngine(language="auto", prefer_gpu=False).recognize(image)
        self.assertTrue(any("chào" in line.text for line in lines), lines)

    def test_auto_recognizes_mixed_scripts_in_one_line(self) -> None:
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if not font_path.is_file():
            self.skipTest("CJK font unavailable")
        image = Image.new("RGB", (1200, 240), "black")
        ImageDraw.Draw(image).text((30, 70), "你好 日本語 Xin chào Hello", fill="white",
                                   font=ImageFont.truetype(str(font_path), 55))
        lines = OCREngine(language="auto", prefer_gpu=False).recognize(image)
        text = " ".join(line.text for line in lines)
        self.assertIn("你好", text)
        self.assertIn("日本語", text)
        self.assertIn("chào", text)

    def test_sparse_subtitles_fall_back_to_full_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = detect_roi(self.video, _SparseDetector({1}), preview_path=Path(directory) / "roi_preview.jpg",
                                sample_frames=4, max_sample_frames=8, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            self.assertEqual(result.roi, FULL_FRAME)
            self.assertEqual(result.mode, "full_frame")
            self.assertEqual(result.frames_checked, 8)
            self.assertTrue(result.preview_path.is_file())

    def test_manual_roi_uses_one_seek(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manual = ROI(0.1, 0.7, 0.9, 0.95)
            result = detect_roi(self.video, _SparseDetector(set()), manual=manual,
                                preview_path=Path(directory) / "roi_preview.jpg",
                                ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            self.assertEqual(result.roi, manual)
            self.assertEqual(result.frames_checked, 1)


if __name__ == "__main__":
    unittest.main()
