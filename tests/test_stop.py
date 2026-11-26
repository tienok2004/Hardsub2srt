"""Stopping mid-OCR saves the subtitles read so far."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core.ocr_engine import RecognizedText, TextBox
from core.pipeline import PipelineConfig, partial_output_path, run_batch
from core.roi_detector import ROI, ROIResult
from core.video_sampler import SampledFrame, VideoMetadata

FPS = 3.0
LINES = ["Xin chào", "Tạm biệt", "Hẹn gặp lại"]  # each line shows for 2 s = 6 samples


class _StopAfter:
    """Fake OCR engine that presses Stop after ``frames`` recognitions."""

    def __init__(self, stop_event: threading.Event, frames: int) -> None:
        self.stop_event, self.frames, self.calls = stop_event, frames, 0

    def recognize(self, frame: SampledFrame) -> list[RecognizedText]:
        self.calls += 1
        if self.calls >= self.frames:
            self.stop_event.set()
        text = LINES[int(frame.timestamp_seconds // 2) % len(LINES)]
        box = TextBox(((10, 10), (300, 10), (300, 60), (10, 60)), 0.99)
        return [RecognizedText(text, 0.99, box)]


class StopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        for name in ("a.mp4", "b.mp4"):
            (self.source / name).touch()
        self.output = self.root / "output"
        self.closed = []

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _frames(self, *_args, **_kwargs):
        try:
            for index in range(int(60 * FPS)):  # a 60-second video
                # Unique pixels per sample so frame skipping never kicks in.
                yield SampledFrame(index / FPS, 4, 2, bytes([index % 256]) * 24)
        finally:
            self.closed.append(True)

    def _run(self, stop_event: threading.Event, engine):
        metadata = lambda video, **_: VideoMetadata(video, 1920, 1080, 60.0, 24.0, None)
        roi = lambda video, _engine, **kw: ROIResult(ROI(0, 0.7, 1, 1), "manual", 0, 0,
                                                    kw["preview_path"], "ROI thủ công")
        config = PipelineConfig(self.source, self.output, fps=FPS, skip_unchanged_frames=False,
                                captions_output="off")  # fake frames are too small to crop
        with patch("core.pipeline.probe_video", metadata), patch("core.pipeline.detect_roi", roi), \
                patch("core.pipeline.sample_frames", self._frames):
            events = []
            summary = run_batch(config, engine=engine, stop_event=stop_event, callback=events.append)
        return summary, events

    def test_stop_saves_partial_srt_and_skips_remaining_videos(self) -> None:
        stop_event = threading.Event()
        engine = _StopAfter(stop_event, frames=15)  # stop after 5 s
        summary, events = self._run(stop_event, engine)

        self.assertTrue(summary.stopped)
        self.assertEqual([item.video.name for item in summary.files], ["a.mp4"])  # b.mp4 not started
        result = summary.files[0]
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.output, partial_output_path(self.output / "a.vi.srt"))
        self.assertEqual(engine.calls, 15)
        self.assertEqual(self.closed, [True])  # decoder closed right away

        text = result.output.read_text(encoding="utf-8-sig")
        self.assertIn("Xin chào", text)
        self.assertIn("Tạm biệt", text)
        self.assertIn("Hẹn gặp lại", text)
        self.assertEqual(result.subtitle_count, 3)
        self.assertFalse((self.output / "a.vi.srt").exists())  # resume will redo this video
        self.assertEqual(events[-1].phase, "stopped")

    def test_stop_before_any_frame_writes_nothing(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        summary, _ = self._run(stop_event, _StopAfter(stop_event, frames=1))
        self.assertEqual(summary.files, ())
        self.assertFalse(self.output.exists() and any(self.output.iterdir()))

    def test_without_stop_writes_normal_srt(self) -> None:
        summary, _ = self._run(threading.Event(), _StopAfter(threading.Event(), frames=10**9))
        self.assertEqual(summary.completed, 2)
        self.assertTrue((self.output / "a.vi.srt").is_file())
        self.assertFalse(partial_output_path(self.output / "a.vi.srt").exists())


if __name__ == "__main__":
    unittest.main()
