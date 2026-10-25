"""Regression checks for the real failure modes, without requiring model outputs."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from mobility.data import OUTPUT, load_data, validate_hourly
from mobility.evaluation import capacity_scenarios, forecast_metrics
from mobility.features import FEATURES, make_features, prediction_inputs, split_by_time
from mobility.scoring import predict_saved


class DataAndFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_data()
        cls.features = make_features(cls.raw)

    def test_source_fingerprint_and_identity(self):
        self.assertEqual(len(self.raw), 17379)
        self.assertTrue((self.raw.casual + self.raw.registered == self.raw.cnt).all())

    def test_bad_values_and_duplicates_are_rejected(self):
        for column, value in [
            ("cnt", -1),
            ("hum", np.nan),
            ("hr", 1.5),
            ("weekday", 9),
        ]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                bad = self.raw.drop(columns="timestamp").copy()
                bad[column] = bad[column].astype(float)
                bad.loc[0, column] = value
                validate_hourly(bad)
        bad = self.raw.drop(columns="timestamp").copy()
        bad.loc[1] = bad.loc[0]
        with self.assertRaises(ValueError):
            validate_hourly(bad)

    def test_lags_match_clock_time_across_missing_hours(self):
        history = self.raw.set_index("timestamp").cnt
        for lag in [24, 168]:
            expected = history.reindex(
                self.features.timestamp - pd.Timedelta(hours=lag)
            )
            np.testing.assert_array_equal(
                expected.to_numpy(), self.features[f"lag_{lag}"]
            )
        self.assertLess(len(self.features), len(self.raw))
        self.assertFalse(self.features.isna().any().any())

    def test_changing_current_and_future_counts_does_not_change_inputs(self):
        cutoff = pd.Timestamp("2012-06-01 12:00")
        changed = self.raw.copy()
        changed.loc[changed.timestamp >= cutoff, "cnt"] += 10000
        features = make_features(changed)
        earlier = self.features.timestamp <= cutoff
        pd.testing.assert_frame_equal(
            self.features.loc[earlier, FEATURES], features.loc[earlier, FEATURES]
        )

    def test_chronological_partitions_cover_eligible_hours_once(self):
        train, validation, test = split_by_time(
            self.features.sample(frac=1, random_state=42)
        )
        self.assertEqual(len(train) + len(validation) + len(test), len(self.features))
        self.assertLess(train.timestamp.max(), validation.timestamp.min())
        self.assertLess(validation.timestamp.max(), test.timestamp.min())
        self.assertLess(train.timestamp.max(), pd.Timestamp("2012-07-01"))
        self.assertGreaterEqual(test.timestamp.min(), pd.Timestamp("2012-10-01"))
        self.assertEqual(len(FEATURES), 8)
        self.assertFalse(
            set(FEATURES) & {"cnt", "casual", "registered", "temp", "instant"}
        )

    def test_scoring_rejects_target_missing_history_and_bad_calendar(self):
        x = self.features[FEATURES].head(1).copy()
        for column in ["cnt", "casual", "registered", "temp"]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                prediction_inputs(x.assign(**{column: 1}))
        with self.assertRaises(ValueError):
            prediction_inputs(x.drop(columns="lag_168"))
        with self.assertRaises(ValueError):
            prediction_inputs(x.assign(lag_24=np.nan))
        with self.assertRaises(ValueError):
            prediction_inputs(x.assign(workingday=1 - x.workingday))

    def test_saved_baseline_uses_the_same_input_contract(self):
        x = self.features[FEATURES].head(2)
        OUTPUT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump({"selected_model": "last_week", "features": FEATURES}, path)
            np.testing.assert_array_equal(predict_saved(x, path), x.lag_168)


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_errors_and_undefined_percentages(self):
        metrics = forecast_metrics([100, 200, 300, 400], [110, 190, 310, 380])
        self.assertEqual(metrics["mae"], 12.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(175))
        self.assertEqual(metrics["wape_pct"], 5)
        self.assertEqual(metrics["bias_pct"], -1)
        zero = forecast_metrics([0], [1])
        self.assertIsNone(zero["wape_pct"])
        self.assertIsNone(zero["bias_pct"])

    def test_invalid_metric_vectors_fail(self):
        for actual, predicted in [
            ([], []),
            ([1], [1, 2]),
            ([1], [np.inf]),
            ([-1], [0]),
        ]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                forecast_metrics(actual, predicted)

    def test_capacity_arithmetic_and_tradeoffs(self):
        table = capacity_scenarios([100], [80])
        first = table.iloc[0]
        self.assertEqual(first.shortage_slots, 16)
        self.assertEqual(first.excess_slots, 0)
        self.assertEqual(first.hypothetical_cost, 48)
        self.assertEqual(first.covered_rentals_pct, 84)
        self.assertTrue(table.shortage_slots.is_monotonic_decreasing)
        self.assertTrue(table.excess_slots.is_monotonic_increasing)
        with self.assertRaises(ValueError):
            capacity_scenarios([100], [80], shortage_cost=-1)

    def test_source_notebooks_have_no_outputs(self):
        for path in Path("notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"], str(path))
                    self.assertEqual(cell["outputs"], [], str(path))


if __name__ == "__main__":
    unittest.main()
