"""Web flow checks with local generated media, without starting a browser."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app import WEB_CACHE, preview_roi, preview_roi_editor, roi_from_image, roi_from_sliders, run_ui


class WebFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.ffmpeg = shutil.which("ffmpeg") or next(
            (str(path) for path in (root / ".tools" / "ffmpeg").rglob("ffmpeg.exe")), None
        )
        if not cls.ffmpeg:
            raise unittest.SkipTest("FFmpeg unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "web_sample.mp4"
        subprocess.run([
            cls.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=size=160x90:rate=30:duration=2", "-c:v", "mpeg4",
            "-y", str(cls.video),
        ], capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_manual_preview_is_available_for_web_output(self) -> None:
        image, status = preview_roi(str(self.video), None, self.temp.name,
                                    False, 0.1, 0.65, 0.9, 0.98)
        self.assertIn("ROI thủ công", status)
        self.assertIsNotNone(image)
        self.assertTrue(Path(image).is_file())
        self.assertEqual(Path(image).parent, WEB_CACHE)

    def test_editor_preview_and_coordinate_sync(self) -> None:
        editor, status, x1, y1, x2, y2 = preview_roi_editor(
            str(self.video), None, self.temp.name, False, 0.1, 0.65, 0.9, 0.98,
        )
        self.assertIn("ROI thủ công", status)
        self.assertTrue(editor["src"].startswith("data:image/jpeg;base64,"))
        self.assertGreater(editor["display_width"], 0)
        self.assertEqual([x1, y1, x2, y2], editor["roi"])

        updated, auto, _ = roi_from_sliders(editor, 0.2, 0.5, 0.8, 0.9)
        self.assertFalse(auto)
        self.assertEqual(updated["roi"], [0.2, 0.5, 0.8, 0.9])
        self.assertEqual(updated["src"], editor["src"])
        self.assertEqual(roi_from_image(updated)[:5], (False, 0.2, 0.5, 0.8, 0.9))

    def test_single_video_flow_returns_download(self) -> None:
        updates = list(run_ui(
            str(self.video), None, self.temp.name, "auto", 3,
            False, False, True, False, 0.1, 0.65, 0.9, 0.98,
            "", False, "", "", "",
        ))
        status, logs, preview, download = updates[-1]
        self.assertIn("1 hoàn thành", status)
        self.assertIn("web_sample.mp4", logs)
        self.assertTrue(Path(preview).is_file())
        self.assertTrue(Path(download).is_file())
        self.assertEqual(Path(download).parent, WEB_CACHE)


if __name__ == "__main__":
    unittest.main()
