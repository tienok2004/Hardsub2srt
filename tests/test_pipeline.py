"""Batch routing checks without running OCR on large media files."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.pipeline import FileResult, PipelineConfig, _as_caption, _near_roi_anchor, run_batch
from core.ocr_engine import RecognizedText, TextBox
from core.roi_detector import FULL_FRAME, ROI, ROIResult
from core.video_sampler import SampledFrame, VideoMetadata


class BatchPipelineTests(unittest.TestCase):
    def test_tall_ocr_box_is_marked_vertical(self) -> None:
        line = RecognizedText("赵丽颖", 0.95,
                              TextBox(((32, 30), (98, 30), (98, 188), (32, 188)), 0.95))
        self.assertTrue(_as_caption(line, (0, 0), 528, 912).vertical)

    def test_bilingual_video_creates_two_language_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "bilingual.mp4"
            video.touch()
            output = root / "output"
            box_zh = TextBox(((40, 130), (160, 130), (160, 150), (40, 150)), 0.99)
            box_en = TextBox(((40, 155), (200, 155), (200, 175), (40, 175)), 0.99)

            class Engine:
                def recognize(self, _frame):
                    return [RecognizedText("我记住了", 0.99, box_zh),
                            RecognizedText("I remember.", 0.99, box_en)]

            def metadata(path, **_):
                return VideoMetadata(path, 240, 200, 1, 24, None)

            def roi(_path, _engine, **kwargs):
                return ROIResult(FULL_FRAME, "manual", 0, 0, kwargs["preview_path"], "ROI thủ công")

            def frames(*_args, **_kwargs):
                for index in range(3):
                    yield SampledFrame(index / 3, 240, 200, bytes([index]) * (240 * 200 * 3))

            config = PipelineConfig(video, output, auto_roi=False, captions_output="off",
                                    skip_unchanged_frames=False)
            with patch("core.pipeline.probe_video", metadata), \
                    patch("core.pipeline.detect_roi", roi), \
                    patch("core.pipeline.sample_frames", frames):
                result = run_batch(config, engine=Engine()).files[0]
            self.assertEqual([path.name for path in result.outputs],
                             ["bilingual.zh.srt", "bilingual.en.srt"])
            self.assertIn("我记住了", result.outputs[0].read_text(encoding="utf-8-sig"))
            self.assertNotIn("I remember.", result.outputs[0].read_text(encoding="utf-8-sig"))
            self.assertIn("I remember.", result.outputs[1].read_text(encoding="utf-8-sig"))

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
            (output / "a.zh.srt").write_text("existing", encoding="utf-8")
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
