"""Behavior checks for subtitle timing, flicker and persistent text."""

import tempfile
import unittest
from pathlib import Path

from core.srt_builder import (
    BuilderConfig, FrameObservation, SubtitleEntry, TextObservation,
    build_subtitles, format_srt, frame_from_ocr, remove_watermarks, write_srt,
)
from core.ocr_engine import RecognizedText, TextBox


def frame(time: float, *texts: TextObservation) -> FrameObservation:
    return FrameObservation(time, texts)


class SubtitleBuilderTests(unittest.TestCase):
    def test_fuzzy_merge_picks_best_confidence_and_formats_srt(self) -> None:
        frames = [
            frame(0.0, TextObservation("Xin chao", 0.60)),
            frame(1 / 3, TextObservation("Xin chào", 0.96)),
            frame(2 / 3, TextObservation("Xin chào", 0.80)),
            frame(1.0, TextObservation("Tạm biệt", 0.91)),
            frame(4 / 3, TextObservation("Tạm biệt", 0.88)),
        ]
        entries = build_subtitles(frames, sample_fps=3)
        self.assertEqual([(entry.id, entry.text) for entry in entries],
                         [(1, "Xin chào"), (2, "Tạm biệt")])
        self.assertEqual((entries[0].start_ms, entries[0].end_ms), (0, 1000))
        self.assertIn("00:00:01,000 --> 00:00:01,667", format_srt(entries))

    def test_watermark_is_removed_without_losing_subtitle(self) -> None:
        watermark = TextObservation("CHANNEL 7", 0.99, (0.01, 0.01, 0.15, 0.06))
        subtitle = TextObservation("Một câu thoại", 0.92, (0.2, 0.85, 0.8, 0.95))
        frames = [frame(float(second), watermark, *((subtitle,) if second in (2, 3) else ()))
                  for second in range(17)]
        entries = build_subtitles(frames, sample_fps=1)
        self.assertEqual([entry.text for entry in entries], ["Một câu thoại"])
        self.assertEqual((entries[0].start_ms, entries[0].end_ms), (2000, 4000))

    def test_single_frame_noise_dropped_and_short_gap_bridged(self) -> None:
        frames = [
            frame(0.0, TextObservation("NOISE", 0.5)),
            frame(0.1),
            frame(0.2, TextObservation("Chào", 0.8)),
            frame(0.3, TextObservation("Chào", 0.9)),
            frame(0.4),
            frame(0.5, TextObservation("Chào", 0.85)),
        ]
        entries = build_subtitles(frames, sample_fps=10)
        self.assertEqual(len(entries), 1)
        self.assertEqual((entries[0].start_ms, entries[0].end_ms), (200, 600))

    def test_writes_unicode_srt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_srt([SubtitleEntry(1, 0, 1000, "Việt Nam 日本語")], Path(directory) / "sub.srt")
            self.assertIn("Việt Nam 日本語", path.read_text(encoding="utf-8-sig"))

    def test_ocr_boxes_are_normalized_for_watermark_tracking(self) -> None:
        line = RecognizedText("Chào", 0.9, TextBox(((20, 80), (120, 80), (120, 95), (20, 95)), 0.9))
        observation = frame_from_ocr(1.5, 200, 100, [line])
        self.assertEqual(observation.texts[0].box, (0.1, 0.8, 0.6, 0.95))

    def test_advertising_url_is_removed_from_subtitle_frame(self) -> None:
        frames = [frame(i / 3, TextObservation("https://hddj008.com", 0.9),
                        TextObservation("我的衣服", 0.9)) for i in range(3)]
        self.assertEqual([entry.text for entry in build_subtitles(frames)], ["我的衣服"])
        cropped_domain = [frame(i / 3, TextObservation("hddj31.co", 0.9),
                                TextObservation("我的衣服", 0.9)) for i in range(3)]
        self.assertEqual([entry.text for entry in build_subtitles(cropped_domain)], ["我的衣服"])

    def test_configured_watermark_term_is_removed(self) -> None:
        frames = [frame(i / 3, TextObservation("红豆短剧", 0.9),
                        TextObservation("我的衣服", 0.9)) for i in range(3)]
        config = BuilderConfig(watermark_terms=("红豆",))
        self.assertEqual([entry.text for entry in build_subtitles(frames, config=config)], ["我的衣服"])

    def test_recurring_moving_logo_and_ocr_fragments_are_removed(self) -> None:
        positions = ((0.1, 0.2, 0.4, 0.25), (0.5, 0.4, 0.8, 0.45),
                     (0.2, 0.7, 0.5, 0.75))
        frames = []
        for second in range(60):
            texts = []
            for start, box in zip((0, 20, 40), positions):
                if start <= second < start + 8:
                    logo = "黄豆短剧"
                    if second == 21:
                        logo = "黄豆短"
                    elif second == 42:
                        logo = "短剧"
                    elif second == 43:
                        logo = "豆"
                    texts.append(TextObservation(logo, 0.9, box))
                    texts.append(TextObservation("H", 0.99,
                                                 (box[0] - 0.04, box[1], box[0] + 0.01, box[3])))
                    texts.append(TextObservation("hddj31.com", 0.99,
                                                 (box[0] + 0.04, box[3], box[2], box[3] + 0.03)))
            if second in (2, 3):
                texts.append(TextObservation("黄豆高中", 0.95, (0.2, 0.8, 0.8, 0.86)))
            if second in (22, 23):
                texts.append(TextObservation("学校举荐", 0.95, (0.2, 0.8, 0.8, 0.86)))
            frames.append(frame(float(second), *texts))
        self.assertEqual([entry.text for entry in build_subtitles(frames, sample_fps=1)],
                         ["黄豆高中", "学校举荐"])
        cleaned = remove_watermarks(frames, sample_fps=1)
        self.assertFalse(any(item.text == "H" for frame_item in cleaned for item in frame_item.texts))
        self.assertEqual([entry.text for entry in build_subtitles(cleaned, sample_fps=1)],
                         ["黄豆高中", "学校举荐"])

    def test_rejects_low_confidence_full_crop_detection(self) -> None:
        bad = RecognizedText("中短剧", 0.75,
                             TextBox(((0, 0), (200, 0), (200, 100), (0, 100)), 0.75))
        partial = RecognizedText("1", 0.62,
                                 TextBox(((80, 0), (200, 0), (200, 100), (80, 100)), 0.62))
        good = RecognizedText("校长", 0.98,
                              TextBox(((50, 60), (150, 60), (150, 80), (50, 80)), 0.98))
        self.assertEqual([item.text for item in frame_from_ocr(1, 200, 100, [bad, partial, good]).texts],
                         ["校长"])


if __name__ == "__main__":
    unittest.main()
