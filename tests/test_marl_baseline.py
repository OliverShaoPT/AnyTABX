from __future__ import annotations

import unittest

from src.baseline.marl_baseline import EarlyStopper, MARLConfig


class EarlyStopperTest(unittest.TestCase):
    def test_stops_after_patience_without_window_improvement(self) -> None:
        config = MARLConfig(
            early_stop_window=2,
            early_stop_patience=2,
            early_stop_min_delta=0.1,
            early_stop_warmup=0,
        )
        stopper = EarlyStopper(config)

        self.assertFalse(stopper.update(1.0, 1)[2])
        self.assertFalse(stopper.update(1.0, 2)[2])
        self.assertFalse(stopper.update(1.0, 3)[2])
        self.assertTrue(stopper.update(1.0, 4)[2])

    def test_disabled_stopper_never_stops(self) -> None:
        config = MARLConfig(
            early_stop_enabled=False,
            early_stop_window=1,
            early_stop_patience=1,
            early_stop_warmup=0,
        )
        stopper = EarlyStopper(config)

        for update in range(1, 5):
            self.assertFalse(stopper.update(0.0, update)[2])


if __name__ == "__main__":
    unittest.main()
