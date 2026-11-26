import unittest

from core.paths import clean_path_text, pick_local_path


class CleanPathTests(unittest.TestCase):
    def win(self, value: str) -> str:
        return clean_path_text(value, windows=True)

    def test_copy_as_path_quotes(self) -> None:
        self.assertEqual(self.win('"C:\\Users\\Admin\\Downloads\\video.mp4"'),
                         "C:\\Users\\Admin\\Downloads\\video.mp4")

    def test_example_from_request(self) -> None:
        self.assertEqual(self.win('"C/Downloads/video.mp4"'), "C/Downloads/video.mp4")

    def test_question_marks_and_mixed_wrappers(self) -> None:
        self.assertEqual(self.win("'“C:\\Phim\\tập 01.mkv?”'"), "C:\\Phim\\tập 01.mkv")
        self.assertEqual(self.win("C:\\Phim\\a*b?.mp4"), "C:\\Phim\\ab.mp4")

    def test_spaces_and_unicode_kept(self) -> None:
        self.assertEqual(self.win('  "D:\\My Videos\\Tiếng Việt\\phim.mp4"  '),
                         "D:\\My Videos\\Tiếng Việt\\phim.mp4")

    def test_invisible_marks_and_extra_lines(self) -> None:
        self.assertEqual(self.win('\u202a"C:\\a.mp4"\u200b\n'), "C:\\a.mp4")
        self.assertEqual(self.win('\n  "C:\\first.mp4"\n"C:\\second.mp4"'), "C:\\first.mp4")

    def test_file_url(self) -> None:
        self.assertEqual(self.win("file:///C:/Users/Admin/My%20Videos/a.mp4"), "C:/Users/Admin/My Videos/a.mp4")
        self.assertEqual(clean_path_text("file:///home/me/a%20b.mp4", windows=False), "/home/me/a b.mp4")

    def test_long_path_prefix(self) -> None:
        self.assertEqual(self.win("\\\\?\\C:\\very\\long.mp4"), "C:\\very\\long.mp4")
        self.assertEqual(self.win("\\\\?\\UNC\\server\\share\\a.mp4"), "\\\\server\\share\\a.mp4")

    def test_network_share_kept(self) -> None:
        self.assertEqual(self.win('"\\\\nas\\phim\\a.mp4"'), "\\\\nas\\phim\\a.mp4")

    def test_linux_keeps_question_mark(self) -> None:
        self.assertEqual(clean_path_text('"/data/what?.mp4"', windows=False), "/data/what?.mp4")

    def test_empty(self) -> None:
        self.assertEqual(clean_path_text(None), "")
        self.assertEqual(clean_path_text('  ""  '), "")


class PickerTests(unittest.TestCase):
    def test_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            pick_local_path("image")


if __name__ == "__main__":
    unittest.main()
