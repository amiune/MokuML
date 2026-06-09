from __future__ import annotations

import unittest

from worker.ml.prepare_table import infer_task_type, prepare_records
from worker.ml.tabpfn_runner import run_tabular_predictions


class MLPreparationTests(unittest.TestCase):
    def test_infers_classification_for_boolean_target(self) -> None:
        self.assertEqual(infer_task_type([True, False, True], "unknown"), "classification")

    def test_prepare_splits_missing_target_rows(self) -> None:
        rows = [
            {"customer_id": "a", "orders": 3, "churned": False},
            {"customer_id": "b", "orders": 1, "churned": True},
            {"customer_id": "c", "orders": 0, "churned": None},
        ]
        prepared = prepare_records(rows, "churned", ["customer_id"])
        self.assertTrue(prepared["ok"])
        self.assertEqual(len(prepared["labeled_rows"]), 2)
        self.assertEqual(len(prepared["prediction_rows"]), 1)

    def test_baseline_runner_returns_predictions_without_optional_dependencies(self) -> None:
        rows = [
            {"customer_id": "a", "orders": 3, "churned": False},
            {"customer_id": "b", "orders": 1, "churned": True},
            {"customer_id": "c", "orders": 4, "churned": False},
            {"customer_id": "d", "orders": 0, "churned": True},
            {"customer_id": "e", "orders": 5, "churned": False},
            {"customer_id": "f", "orders": 2, "churned": None},
        ]
        result = run_tabular_predictions(rows, "churned", ["customer_id"], "classification")
        self.assertTrue(result["ok"])
        self.assertTrue(result["predictions"])

    def test_sklearn_runner_handles_imbalanced_validation_split(self) -> None:
        rows = [
            {"customer_id": f"c{i}", "orders": i % 4, "churned": i == 0}
            for i in range(12)
        ]
        result = run_tabular_predictions(rows, "churned", ["customer_id"], "classification")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"]["checkpoint"], "local sklearn")
        self.assertNotIn("dependency-free baseline", " ".join(result["warnings"]))

    def test_sklearn_runner_handles_no_feature_columns(self) -> None:
        rows = [
            {"customer_id": f"c{i}", "churned": i % 2 == 0}
            for i in range(10)
        ]
        result = run_tabular_predictions(rows, "churned", ["customer_id"], "classification")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"]["checkpoint"], "local sklearn")
        self.assertTrue(result["predictions"])

    def test_sklearn_runner_handles_mixed_categorical_values(self) -> None:
        rows = [
            {
                "customer_id": f"c{i}",
                "segment": "retail" if i % 2 else 10,
                "metadata": {"tier": i % 3},
                "orders": str(i),
                "churned": i % 3 == 0,
            }
            for i in range(12)
        ]
        result = run_tabular_predictions(rows, "churned", ["customer_id"], "classification")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"]["checkpoint"], "local sklearn")
        self.assertNotIn("dependency-free baseline", " ".join(result["warnings"]))

    def test_sklearn_runner_returns_metrics_with_prediction_rows(self) -> None:
        rows = [
            {"customer_id": f"c{i}", "orders": i % 5, "churned": i % 3 == 0}
            for i in range(20)
        ]
        rows.extend(
            [
                {"customer_id": "future-1", "orders": 2, "churned": None},
                {"customer_id": "future-2", "orders": 4, "churned": None},
            ]
        )
        result = run_tabular_predictions(rows, "churned", ["customer_id"], "classification")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"]["checkpoint"], "local sklearn")
        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(result["metrics"]["validation_rows"], 4)
        self.assertEqual(len(result["predictions"]), 2)

    def test_sklearn_runner_returns_regression_metrics_with_prediction_rows(self) -> None:
        rows = [
            {"quote_id": f"q{i}", "line_items": i % 5, "expected_value": float(i * 10)}
            for i in range(20)
        ]
        rows.append({"quote_id": "future-1", "line_items": 3, "expected_value": None})
        result = run_tabular_predictions(rows, "expected_value", ["quote_id"], "regression")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"]["checkpoint"], "local sklearn")
        self.assertIn("rmse", result["metrics"])
        self.assertIn("mae", result["metrics"])
        self.assertEqual(len(result["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
