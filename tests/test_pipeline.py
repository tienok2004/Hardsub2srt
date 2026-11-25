"""Batch routing checks without running OCR on large media files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.pipeline import FileResult, PipelineConfig, _near_roi_anchor, run_batch
from core.ocr_engine import RecognizedText, TextBox
from core.roi_detector import ROI


class BatchPipelineTests(unittest.TestCase):
    def test_logo_adjacent_to_subtitle_is_excluded(self) -> None:
        logo = RecognizedText("红豆短剧", 0.95,
                              TextBox(((50, 20), (290, 20), (290, 100), (50, 100)), 0.95))
        subtitle = RecognizedText("我的衣服", 0.95,
                                  TextBox(((293, 65), (511, 65), (511, 130), (293, 130)), 0.95))
        roi = ROI(0.129375, 0.7298, 0.870625, 0.8671)
        self.assertEqual(_near_roi_anchor([logo, subtitle], roi, 800), [subtitle])

    def test_same_stem_collision_skips_second_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (source / "a.mp4").touch()
            (source / "a.mkv").touch()
            output = root / "output"
            with patch("core.pipeline._process_one") as process:
                process.side_effect = lambda video, destination, *_: FileResult(video, destination, "completed")
                summary = run_batch(PipelineConfig(source, output), engine=object())
            self.assertEqual(summary.completed, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.files[1].status, "skipped_collision")
            self.assertEqual(process.call_count, 1)

    def test_existing_srt_is_skipped_before_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "a.mp4"
            video.touch()
            output = root / "output"
            output.mkdir()
            (output / "a.srt").write_text("existing", encoding="utf-8")
            with patch("core.pipeline._process_one") as process:
                summary = run_batch(PipelineConfig(video, output), engine=object())
            self.assertEqual(summary.files[0].status, "skipped_existing")
            process.assert_not_called()

    def test_bad_video_does_not_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (source / "a.mp4").touch()
            (source / "b.mp4").touch()
            with patch("core.pipeline._process_one") as process:
                def side_effect(video, output, *_):
                    if video.name == "a.mp4":
                        raise RuntimeError("corrupt video")
                    return FileResult(video, output, "completed")
                process.side_effect = side_effect
                summary = run_batch(PipelineConfig(source, root / "output"), engine=object())
            self.assertEqual((summary.failed, summary.completed), (1, 1))
            self.assertEqual(process.call_count, 2)

    def test_recursive_preserves_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            (source / "season1").mkdir(parents=True)
            (source / "season1" / "a.mp4").touch()
            with patch("core.pipeline._process_one") as process:
                process.side_effect = lambda video, destination, *_: FileResult(video, destination, "completed")
                summary = run_batch(PipelineConfig(source, root / "output", recursive=True,
                                                   preserve_subfolders=True), engine=object())
            self.assertEqual(summary.files[0].output, root / "output" / "season1" / "a.srt")


if __name__ == "__main__":
    unittest.main()
