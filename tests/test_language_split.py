import tempfile
import unittest
from pathlib import Path

from core.language_split import language_of, split_entries, split_observations
from core.pipeline import _write_outputs
from core.srt_builder import FrameObservation, SubtitleEntry, TextObservation, build_subtitles


class LanguageSplitTests(unittest.TestCase):
    def test_script_and_vietnamese_diacritics(self):
        self.assertEqual(language_of("我记住了"), "zh")
        self.assertEqual(language_of("覚えました"), "ja")
        self.assertEqual(language_of("Tôi nhớ rồi"), "vi")
        self.assertEqual(language_of("I remember."), "en")
        self.assertIsNone(language_of("22"))
        self.assertIsNone(language_of(r"{\an8}(22)"))

    def test_bilingual_lines_in_same_frame_keep_their_own_timestamps(self):
        frames = [FrameObservation(index / 3, (
            TextObservation("我记住了", 0.99),
            TextObservation("I remember.", 0.99),
        )) for index in range(3)]
        tracks, primary = split_observations(frames)
        self.assertEqual(primary, "zh")
        self.assertEqual(set(tracks), {"zh", "en"})
        subtitles = {language: build_subtitles(items) for language, items in tracks.items()}
        self.assertEqual([item.text for item in subtitles["zh"]], ["我记住了"])
        self.assertEqual([item.text for item in subtitles["en"]], ["I remember."])
        self.assertEqual(subtitles["zh"][0].start_ms, subtitles["en"][0].start_ms)
        captions = split_entries([SubtitleEntry(1, 0, 1000, "(恭喜，终于等到这一天了。)")],
                                 primary=primary)
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_outputs(subtitles, captions, Path(directory) / "1.srt", "merged", primary)
            self.assertEqual([path.name for path in paths], ["1.zh.srt", "1.en.srt"])
            zh = paths[0].read_text(encoding="utf-8-sig")
            en = paths[1].read_text(encoding="utf-8-sig")
            self.assertIn("我记住了", zh)
            self.assertIn("(恭喜，终于等到这一天了。)", zh)
            self.assertNotIn("I remember.", zh)
            self.assertIn("I remember.", en)
            self.assertNotIn("我记住了", en)

    def test_unknown_numbers_follow_primary_language(self):
        frames = [FrameObservation(index / 3, (
            TextObservation("你好", 0.99), TextObservation("Hello", 0.99),
            TextObservation("22", 0.99),
        )) for index in range(3)]
        tracks, primary = split_observations(frames)
        self.assertEqual(primary, "zh")
        self.assertIn("22", [item.text for item in tracks["zh"][0].texts])
        self.assertNotIn("22", [item.text for item in tracks["en"][0].texts])
        captions = split_entries([SubtitleEntry(1, 0, 1000, r"{\an8}(22)")], primary=primary)
        self.assertEqual(set(captions), {"zh"})

    def test_empty_video_has_undetermined_filename(self):
        tracks, primary = split_observations([FrameObservation(0.0, ())])
        self.assertEqual(primary, "und")
        self.assertEqual(set(tracks), {"und"})
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_outputs({"und": []}, {}, Path(directory) / "empty.srt", "merged", primary)
            self.assertEqual(paths[0].name, "empty.und.srt")


if __name__ == "__main__":
    unittest.main()
