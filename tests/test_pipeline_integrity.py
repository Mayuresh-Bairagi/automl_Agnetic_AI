import os
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.Classifier.MLClassifier import AutoMLClassifier
from src.Regression.regression import AutoMLRegressor


class PipelineIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        self.session_dir = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", self.session_id
        )
        os.makedirs(self.session_dir, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def _write_processed(self, df: pd.DataFrame) -> None:
        df.to_csv(os.path.join(self.session_dir, "processed_file.csv"), index=False)

    def test_classifier_init_does_not_call_feature_selector(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4],
                "target": [0, 1, 0, 1],
            }
        )
        self._write_processed(df)

        result = {"target_variable": "target", "problem_type": "classification"}
        with patch("src.problem_statement.AutoFeatureSelector.FeatureSelector.__init__", side_effect=AssertionError("Should not run in __init__")):
            model = AutoMLClassifier(
                session_id=self.session_id,
                problem_statement="predict target",
                result=result,
                df=df,
            )

        self.assertEqual(model.target_col, "target")
        self.assertEqual(model.dropped_features, [])

    def test_classifier_preprocess_removes_duplicates_and_missing_target(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1, 1, 2, 3],
                "f2": ["A", "A", "B", "C"],
                "target": [1, 1, 0, np.nan],
            }
        )
        self._write_processed(df)

        result = {"target_variable": "target", "problem_type": "classification"}
        model = AutoMLClassifier(
            session_id=self.session_id,
            problem_statement="predict target",
            result=result,
            df=df,
        )

        X, y = model.preprocess()
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)
        self.assertNotIn("target", X.columns)

    def test_classifier_preprocess_encodes_string_target(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4],
                "f2": ["A", "B", "A", "C"],
                "target": ["Yes", "No", "Yes", "No"],
            }
        )
        self._write_processed(df)

        result = {"target_variable": "target", "problem_type": "classification"}
        model = AutoMLClassifier(
            session_id=self.session_id,
            problem_statement="predict target",
            result=result,
            df=df,
        )

        _, y = model.preprocess()
        self.assertTrue(np.issubdtype(np.array(y).dtype, np.number))

    def test_train_only_dropped_features_excludes_target(self) -> None:
        df = pd.DataFrame(
            {
                "leaky_col": [10, 20, 30, 40],
                "signal": [1, 0, 1, 0],
                "target": [0, 1, 0, 1],
            }
        )
        self._write_processed(df)

        result = {"target_variable": "target", "problem_type": "classification"}
        model = AutoMLClassifier(
            session_id=self.session_id,
            problem_statement="predict target",
            result=result,
            df=df,
        )

        X_train = pd.DataFrame({"leaky_col": [1, 2], "signal": [0, 1]})
        y_train = pd.Series([0, 1])

        with patch(
            "src.problem_statement.AutoFeatureSelector.FeatureSelector.__init__",
            return_value=None,
        ), patch(
            "src.problem_statement.AutoFeatureSelector.FeatureSelector.llm_response",
            return_value={"dropped_features": ["leaky_col", "target"]},
        ):
            dropped = model._resolve_train_only_dropped_features(X_train, y_train)

        self.assertIn("leaky_col", dropped)
        self.assertNotIn("target", dropped)

    def test_regressor_preprocessor_handles_unseen_categories(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A", "B", "A", "B"],
                "num": [1.0, 2.5, 3.1, 4.8],
                "target": [10.0, 20.0, 30.0, 40.0],
            }
        )
        self._write_processed(df)

        result = {"target_variable": "target", "problem_type": "regression"}
        model = AutoMLRegressor(
            session_id=self.session_id,
            problem_statement="predict target",
            result=result,
            df=df,
        )

        X_train = pd.DataFrame({"city": ["A", "B"], "num": [1.0, 2.0]})
        X_test = pd.DataFrame({"city": ["C"], "num": [3.0]})

        X_train_t, _, _ = model._fit_preprocessor(X_train)
        X_test_t = model._transform_with_preprocessor(X_test)

        self.assertEqual(X_train_t.shape[1], X_test_t.shape[1])


if __name__ == "__main__":
    unittest.main()
