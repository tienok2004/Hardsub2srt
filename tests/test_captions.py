import unittest

from core.captions import (
    TOP_TAG, CaptionConfig, CaptionFrame, CaptionText, ScreenTextConfig, angle_of,
    build_captions, build_screen_text, caption_config_from, merge_entries,
    screen_text_config_from,
)
from core.pipeline import _caption_zones, _without_screen_region
from core.roi_detector import ROI
from core.srt_builder import FrameObservation, SubtitleEntry, TextObservation

INTERVAL = 1 / 3
DIALOGUE_HEIGHT = 0.06


def _text(text, x=0.40, y=0.08, *, width=0.20, height=0.05, confidence=0.95,
          angle=0.0, vertical=False):
    return CaptionText(text, confidence, (x, y, x + width, y + height), angle, vertical)


def _frames(duration, texts_at, total=None):
    """Samples every INTERVAL seconds; texts_at(t) gives the texts on screen at t."""
    count = round((total if total is not None else duration) / INTERVAL)
    return [CaptionFrame(i * INTERVAL, tuple(texts_at(i * INTERVAL))) for i in range(count)]


def _build(frames, **kwargs):
    return build_captions(frames, interval=INTERVAL, dialogue_height=DIALOGUE_HEIGHT, **kwargs)


class CaptionFilterTests(unittest.TestCase):
    def test_groups_vertical_character_name_and_description(self):
        frames = _frames(5, lambda t: [
            _text("赵丽颖", x=0.06, y=0.04, width=0.12, height=0.18,
                  confidence=0.88, vertical=True),
            _text("金鸡奖影后", x=0.19, y=0.09, width=0.10, height=0.22,
                  confidence=0.99, vertical=True),
            _text("hddjai.cc", x=0.50, y=0.12, width=0.08, height=0.20,
                  vertical=True),
        ] if 1 <= t < 3.5 else [])
        entries, rejected = _build(frames)
        self.assertEqual([entry.text for entry in entries],
                         [TOP_TAG + "(赵丽颖：金鸡奖影后)"])
        self.assertIn("chữ dọc không phải chú thích", rejected)

    def test_vertical_ocr_variants_at_same_position_do_not_repeat_name(self):
        def texts(t):
            if not 1 <= t < 4:
                return []
            correct = round(t / INTERVAL) % 2 == 0
            return [
                _text("赵丽颖" if correct else "赵丽颍", x=0.06, y=0.04,
                      width=0.12, height=0.18, confidence=0.95 if correct else 0.85,
                      vertical=True),
                _text("金鸡奖影后", x=0.19, y=0.09, width=0.10, height=0.22,
                      vertical=True),
            ]
        entries, _ = _build(_frames(5, texts))
        self.assertEqual([entry.text for entry in entries],
                         [TOP_TAG + "(赵丽颖：金鸡奖影后)"])

    def test_keeps_a_level_still_caption_and_tags_it_for_the_top(self):
        frames = _frames(8, lambda t: [_text("Hà Nội, mùa thu năm 1975")] if 1 <= t < 5 else [])
        entries, rejected = _build(frames)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, TOP_TAG + "(Hà Nội, mùa thu năm 1975)")
        self.assertAlmostEqual(entries[0].start_ms, 1000, delta=40)
        self.assertAlmostEqual(entries[0].end_ms, 5000, delta=40)
        self.assertEqual(rejected, {})

    def test_drops_a_logo_that_never_leaves(self):
        frames = _frames(30, lambda t: [_text("PhimHay TV", x=0.85, width=0.1)])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("đứng yên quá lâu (logo)", rejected)

    def test_drops_an_animated_logo_that_keeps_coming_back(self):
        frames = _frames(30, lambda t: [_text("Kênh Giải Trí 24h", x=0.02)] if t % 6 < 3 else [])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("lặp lại nhiều lần (logo động)", rejected)

    def test_drops_ads_and_user_watermarks(self):
        frames = _frames(10, lambda t: [_text("Xem full tại xemphimnhanh.net", y=0.20),
                                        _text("Bản dịch bởi NhómABC", y=0.02)] if 2 <= t < 6 else [])
        entries, rejected = _build(frames, watermark_terms=["NhómABC"])
        self.assertEqual(entries, [])
        self.assertEqual(rejected.get("quảng cáo / watermark"), 2)

    def test_drops_short_lived_corner_logo(self):
        frames = _frames(8, lambda t: [_text("红豆短剧", x=0.16, y=0.05, width=0.22)]
                         if 1 <= t < 5 else [])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("logo ở góc khung hình", rejected)

    def test_keeps_long_editorial_caption_in_a_corner(self):
        frames = _frames(8, lambda t: [_text("Hà Nội, 1975", x=0.05, y=0.05, width=0.30)]
                         if 1 <= t < 5 else [])
        entries, _ = _build(frames)
        self.assertEqual([entry.text for entry in entries], [TOP_TAG + "(Hà Nội, 1975)"])

    def test_drops_tilted_scene_text(self):
        frames = _frames(8, lambda t: [_text("ADIDAS ORIGINALS", angle=-12)] if 1 <= t < 5 else [])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("chữ nghiêng (biển hiệu, áo quần)", rejected)

    def test_drops_a_sign_that_moves_with_the_camera(self):
        frames = _frames(8, lambda t: [_text("CỬA HÀNG HOA MAI", x=0.20 + 0.01 * t)] if 1 <= t < 6 else [])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("di chuyển theo cảnh", rejected)

    def test_drops_fine_print_and_low_confidence_text(self):
        frames = _frames(8, lambda t: [
            _text("Bản quyền thuộc về đơn vị phát hành", y=0.25, height=0.012),
            _text("Chữ mờ", y=0.02, confidence=0.5),
        ] if 1 <= t < 5 else [])
        entries, rejected = _build(frames)
        self.assertEqual(entries, [])
        self.assertIn("cỡ chữ bất thường", rejected)
        self.assertIn("độ tin cậy thấp", rejected)

    def test_ignores_flashes_shorter_than_the_minimum(self):
        frames = _frames(5, lambda t: [_text("Nháy")] if 1 <= t < 1.5 else [])
        self.assertEqual(_build(frames)[0], [])

    def test_survives_one_missed_sample(self):
        def texts(t):
            if 1 <= t < 5 and abs(t - 3) > 0.1:
                return [_text("Ba năm sau")]
            return []
        entries, _ = _build(_frames(8, texts))
        self.assertEqual(len(entries), 1)

    def test_no_top_tag_for_captions_in_the_lower_half(self):
        frames = _frames(6, lambda t: [_text("Ghi chú", y=0.60)] if 1 <= t < 4 else [])
        entries, _ = _build(frames)
        self.assertEqual(entries[0].text, "(Ghi chú)")


class CaptionHelpersTests(unittest.TestCase):
    def test_angle_of_quad(self):
        self.assertAlmostEqual(angle_of([(0, 0), (10, 0), (10, 2), (0, 2)]), 0.0)
        self.assertAlmostEqual(angle_of([(0, 0), (10, 10)]), 45.0)

    def test_merge_orders_by_start_and_renumbers(self):
        dialogue = [SubtitleEntry(1, 0, 2000, "A"), SubtitleEntry(2, 3000, 4000, "B")]
        captions = [SubtitleEntry(1, 1000, 5000, TOP_TAG + "C")]
        merged = merge_entries(dialogue, captions)
        self.assertEqual([(e.id, e.text) for e in merged], [(1, "A"), (2, TOP_TAG + "C"), (3, "B")])

    def test_zones_cut_out_the_dialogue_band(self):
        dialogue = ROI(0.1, 0.80, 0.9, 0.95)
        self.assertEqual(_caption_zones(dialogue, ((0.0, 0.0, 1.0, 0.35),)), [ROI(0.0, 0.0, 1.0, 0.35)])
        # A zone overlapping the dialogue keeps its larger free part.
        self.assertEqual(_caption_zones(dialogue, ((0.0, 0.5, 1.0, 1.0),)), [ROI(0.0, 0.5, 1.0, 0.79)])
        # Nothing usable left.
        self.assertEqual(_caption_zones(dialogue, ((0.0, 0.78, 1.0, 1.0),)), [])

    def test_config_from_settings(self):
        config = caption_config_from({"max_repeats": 4, "zones": [[0, 0, 1, 0.3]]}, watermark_seconds=20)
        self.assertEqual(config.max_repeats, 4)
        self.assertEqual(config.max_seconds, 20)
        self.assertEqual(config.zones, ((0.0, 0.0, 1.0, 0.3),))
        self.assertEqual(caption_config_from(None), CaptionConfig())

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            CaptionConfig(zones=((0.5, 0.0, 0.2, 0.3),))
        with self.assertRaises(ValueError):
            CaptionConfig(max_repeats=1)


class ScreenTextTests(unittest.TestCase):
    def test_site_name_beside_domain_is_not_a_screen_message(self):
        frames = _frames(5, lambda t: [
            _text("H薰豆短居", x=0.35, y=0.43, width=0.3, height=0.06),
            _text("hddj30.com", x=0.44, y=0.49, width=0.2, height=0.03),
        ] if 1 <= t < 3 else [])
        entries, reasons = build_screen_text(frames, interval=INTERVAL)
        self.assertEqual(entries, [])
        self.assertIn("logo cạnh địa chỉ web", reasons)

    def test_phone_message_is_parenthesized_while_logos_and_headers_are_ignored(self):
        def texts(t):
            result = [_text("xunse.org", x=0.02, y=0.01, height=0.035),
                      _text("红豆短剧", x=0.14, y=0.05),
                      _text("红豆短剧", x=0.73, y=0.93)]
            if 1 <= t < 5:
                result.extend([_text("消息", x=0.65, y=0.39, width=0.07),
                               _text("21.38", x=0.57, y=0.36, width=0.05),
                               _text("恭喜，终于等到这一天了。", x=0.52 + 0.004 * t,
                                     y=0.43 + 0.003 * t, width=0.27,
                                     height=0.06, angle=13)])
            return result
        entries, _ = build_screen_text(_frames(8, texts), interval=INTERVAL)
        self.assertEqual([entry.text for entry in entries], ["(恭喜，终于等到这一天了。)"])
        self.assertAlmostEqual(entries[0].start_ms, 1000, delta=40)
        self.assertAlmostEqual(entries[0].end_ms, 5000, delta=40)

    def test_rejects_repeated_ad_and_static_brand_inside_screen_area(self):
        def texts(t):
            result = [_text("xunse.org", x=0.4, y=0.4)] if t < 4 else []
            if t < 20:
                result.append(_text("红豆短剧官方出品", x=0.45, y=0.5))
            return result
        entries, reasons = build_screen_text(_frames(25, texts), interval=INTERVAL)
        self.assertEqual(entries, [])
        self.assertIn("quảng cáo / watermark", reasons)
        self.assertIn("đứng yên quá lâu (logo)", reasons)

    def test_full_frame_dialogue_does_not_repeat_accepted_phone_content(self):
        zones = [ROI(0.12, 0.28, 0.88, 0.78)]
        frame = FrameObservation(1.0, (
            TextObservation("消息", 0.99, (0.65, 0.39, 0.72, 0.42)),
            TextObservation("恭喜，终于等到这一天了。", 0.99, (0.52, 0.43, 0.79, 0.49)),
            TextObservation("Lời thoại", 0.99, (0.3, 0.84, 0.7, 0.9)),
        ))
        result = _without_screen_region([frame], [SubtitleEntry(1, 1000, 2000, "(恭喜，终于等到这一天了。)")], zones)
        self.assertEqual([text.text for text in result[0].texts], ["Lời thoại"])

    def test_screen_configuration(self):
        config = screen_text_config_from({"enabled": False, "zones": [[0.2, 0.3, 0.8, 0.7]]})
        self.assertFalse(config.enabled)
        self.assertEqual(config.zones, ((0.2, 0.3, 0.8, 0.7),))
        with self.assertRaises(ValueError):
            ScreenTextConfig(zones=((0.9, 0.2, 0.1, 0.8),))


if __name__ == "__main__":
    unittest.main()
