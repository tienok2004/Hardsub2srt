import unittest

from core.pipeline import EtaEstimator, format_clock


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class EtaTests(unittest.TestCase):
    def test_unknown_until_enough_time_has_passed(self) -> None:
        clock = _Clock()
        eta = EtaEstimator(900, clock=clock)
        clock.now = 1.0
        self.assertIsNone(eta.update(10))

    def test_steady_rate(self) -> None:
        clock = _Clock()
        eta = EtaEstimator(900, clock=clock)
        for second in range(1, 11):
            clock.now = float(second)
            left = eta.update(second * 10)  # 10 frames/s
        self.assertAlmostEqual(left, 80.0)  # 800 frames left at 10 frames/s

    def test_follows_recent_speed_change(self) -> None:
        clock = _Clock()
        eta = EtaEstimator(10_000, clock=clock, window_seconds=30)
        done = 0
        for second in range(1, 61):  # one minute at 10 frames/s
            clock.now = float(second)
            done += 10
            eta.update(done)
        for second in range(61, 121):  # then a quiet scene at 40 frames/s
            clock.now = float(second)
            done += 40
            left = eta.update(done)
        # Whole-run average would be 25 frames/s; the window sees ~40 frames/s.
        self.assertAlmostEqual(left, (10_000 - done) / 40, delta=5)

    def test_first_call_warmup_is_excluded(self) -> None:
        clock = _Clock()
        eta = EtaEstimator(1000, clock=clock)
        clock.now = 2.0  # first frame took 2 s (GPU warm-up)
        self.assertIsNone(eta.update(1))
        for second in range(3, 9):
            clock.now = float(second)
            left = eta.update(1 + (second - 2) * 20)  # then a steady 20 frames/s
        self.assertAlmostEqual(left, (1000 - 121) / 20)

    def test_never_negative(self) -> None:
        clock = _Clock()
        eta = EtaEstimator(100, clock=clock)
        clock.now = 1.0
        eta.update(10)
        clock.now = 5.0
        self.assertEqual(eta.update(120), 0)

    def test_format_clock(self) -> None:
        self.assertEqual(format_clock(0), "0:00:00")
        self.assertEqual(format_clock(75.4), "0:01:15")
        self.assertEqual(format_clock(3 * 3600 + 62), "3:01:02")


if __name__ == "__main__":
    unittest.main()
