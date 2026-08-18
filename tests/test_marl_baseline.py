from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.baseline.marl_baseline import EarlyStopper, MARLConfig, MetricRecorder


class EarlyStopperTest(unittest.TestCase):
    def test_stops_after_patience_without_window_improvement(self) -> None:
        config = MARLConfig(
            early_stop_window=2,
            early_stop_patience=2,
            early_stop_min_delta=0.1,
            early_stop_warmup=0,
        )
        stopper = EarlyStopper(config)

        self.assertFalse(stopper.update(1.0, 1).triggered)
        self.assertFalse(stopper.update(1.0, 2).triggered)
        self.assertFalse(stopper.update(1.0, 3).triggered)
        triggered = stopper.update(1.0, 4)
        self.assertTrue(triggered.triggered)
        self.assertTrue(triggered.would_stop)
        self.assertEqual(stopper.trigger_count, 1)
        self.assertFalse(stopper.update(1.0, 5).triggered)

    def test_disabled_stopper_never_stops(self) -> None:
        config = MARLConfig(
            early_stop_enabled=False,
            early_stop_window=1,
            early_stop_patience=1,
            early_stop_warmup=0,
        )
        stopper = EarlyStopper(config)

        for update in range(1, 5):
            self.assertFalse(stopper.update(0.0, update).would_stop)

    def test_metric_recorder_flushes_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            recorder = MetricRecorder(path)
            recorder.write({"update_steps": 1, "episode_returns": 0.25})
            recorder.close()

            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["update_steps"], "1")
            self.assertEqual(rows[0]["episode_returns"], "0.25")


if __name__ == "__main__":
    unittest.main()
