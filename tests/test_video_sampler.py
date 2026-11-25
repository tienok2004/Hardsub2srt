"""Integration checks against a generated, real video container."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.video_sampler import VideoError, probe_video, sample_frames


class VideoSamplerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = os.getenv("HARDSUB_FFMPEG") or shutil.which("ffmpeg")
        cls.ffprobe = os.getenv("HARDSUB_FFPROBE") or shutil.which("ffprobe")
        local_tools = Path(__file__).resolve().parents[1] / ".tools" / "ffmpeg"
        if not cls.ffmpeg and local_tools.is_dir():
            cls.ffmpeg = next((str(path) for path in local_tools.rglob("ffmpeg.exe")), None)
        if not cls.ffprobe and local_tools.is_dir():
            cls.ffprobe = next((str(path) for path in local_tools.rglob("ffprobe.exe")), None)
        if not cls.ffmpeg or not cls.ffprobe:
            raise unittest.SkipTest("FFmpeg/ffprobe unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "clip.mp4"
        subprocess.run(
            [cls.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=2",
             "-c:v", "mpeg4", "-y", str(cls.video)],
            capture_output=True, check=True, timeout=30,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_metadata_and_sample_timestamps(self) -> None:
        info = probe_video(self.video, ffprobe=self.ffprobe)
        self.assertEqual((info.width, info.height), (160, 90))
        self.assertAlmostEqual(info.duration_seconds, 2.0, delta=0.1)
        frames = list(sample_frames(self.video, fps=3, ffmpeg=self.ffmpeg, metadata=info))
        self.assertEqual(len(frames), 6)
        self.assertEqual([round(f.timestamp_seconds, 3) for f in frames],
                         [0.0, 0.333, 0.667, 1.0, 1.333, 1.667])
        self.assertTrue(all(len(frame.rgb24) == 160 * 90 * 3 for frame in frames))

    def test_invalid_video_is_reported(self) -> None:
        bad = Path(self.temp.name) / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(VideoError):
            probe_video(bad, ffprobe=self.ffprobe)

    def test_ffmpeg_crops_before_transferring_frames(self) -> None:
        frame = next(sample_frames(self.video, fps=3, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe,
                                   crop=(10, 20, 110, 70)))
        self.assertEqual((frame.width, frame.height, len(frame.rgb24)), (100, 50, 15_000))


if __name__ == "__main__":
    unittest.main()
