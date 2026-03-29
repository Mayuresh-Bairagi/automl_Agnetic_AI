import io
import os
import shutil
import sys
import types
import unittest
import uuid
import zipfile
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn as sklearn_pkg


if "ydata_profiling" not in sys.modules:
    fake_ydata = types.ModuleType("ydata_profiling")

    class _DummyProfileReport:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def to_file(self, output_file: str) -> None:
            with open(output_file, "w", encoding="utf-8") as handle:
                handle.write("<html><body>dummy</body></html>")

    setattr(fake_ydata, "ProfileReport", _DummyProfileReport)
    sys.modules["ydata_profiling"] = fake_ydata

from app.main import app
from src.inference.model_explainer import build_best_model_summary
from src.session_tracking.preprocessing_tracker import validate_preprocessing_artifact


class InferenceTrackingEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        self.legacy_session_id = f"test_legacy_session_{uuid.uuid4().hex[:8]}"
        self.session_dir = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", self.session_id
        )
        self.legacy_session_dir = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", self.legacy_session_id
        )
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.legacy_session_dir, exist_ok=True)
        self.client = TestClient(app)

        train_df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0],
                "cat": ["A", "B", "A", "B"],
            }
        )
        y = np.array([10.0, 20.0, 30.0, 40.0])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), ["num"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat"]),
            ],
            remainder="drop",
        )
        x_t = preprocessor.fit_transform(train_df)

        model = LinearRegression()
        model.fit(x_t, y)

        preprocessing_obj = {
            "preprocessor": preprocessor,
            "target_encoder": None,
            "dropped_features": [],
            "pruned_features": {
                "high_missing": [],
                "constant": [],
                "high_cardinality": [],
            },
            "tracking_metadata": {
                "session_id": self.session_id,
                "problem_type": "regression",
                "target_column": "target",
                "feature_names_in": list(getattr(preprocessor, "feature_names_in_", [])),
                "dropped_features": [],
                "library_versions": {
                    "python": "3.13.0",
                    "sklearn": sklearn_pkg.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            },
        }

        joblib.dump(
            preprocessing_obj,
            os.path.join(self.session_dir, "preprocessing.joblib"),
        )
        joblib.dump(
            model,
            os.path.join(self.session_dir, "LinearRegression.joblib"),
        )

        # Legacy-style artifact: valid in compatibility mode, invalid in strict mode
        legacy_preprocessing_obj = {
            "preprocessor": preprocessor,
            "target_encoder": None,
            "dropped_features": [],
            "pruned_features": {
                "high_missing": [],
                "constant": [],
                "high_cardinality": [],
            },
        }
        joblib.dump(
            legacy_preprocessing_obj,
            os.path.join(self.legacy_session_dir, "preprocessing.joblib"),
        )
        joblib.dump(
            model,
            os.path.join(self.legacy_session_dir, "LinearRegression.joblib"),
        )

    def tearDown(self) -> None:
        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir, ignore_errors=True)
        if os.path.exists(self.legacy_session_dir):
            shutil.rmtree(self.legacy_session_dir, ignore_errors=True)

    def test_preprocessing_validation_endpoint_returns_valid(self) -> None:
        response = self.client.get(
            f"/session/{self.session_id}/preprocessing/validate"
        )
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn("validation", payload)
        self.assertIn("strict_validation", payload)
        self.assertTrue(payload["validation"].get("valid"))
        self.assertTrue(payload["strict_validation"].get("valid"))
        self.assertIn("runtime_versions", payload["validation"])

    def test_model_usage_script_zip_endpoint_returns_expected_files(self) -> None:
        response = self.client.get(
            f"/model-usage-script/{self.session_id}/LinearRegression.joblib"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/zip")

        zip_bytes = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_bytes, "r") as zipf:
            names = set(zipf.namelist())

        self.assertIn("use_model.py", names)
        self.assertIn("requirements.txt", names)
        self.assertIn("README.md", names)
        self.assertIn("TECHNICAL_NOTES.md", names)
        self.assertIn("LinearRegression.joblib", names)
        self.assertIn("preprocessing.joblib", names)

    def test_model_usage_script_zip_allows_legacy_override(self) -> None:
        response = self.client.get(
            f"/model-usage-script/{self.legacy_session_id}/LinearRegression.joblib"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("x-preprocessing-validation-mode"),
            "legacy-override",
        )

    def test_model_usage_script_zip_blocks_legacy_when_override_disabled(self) -> None:
        response = self.client.get(
            f"/model-usage-script/{self.legacy_session_id}/LinearRegression.joblib?allow_legacy_override=false"
        )
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        detail = payload.get("detail", {})
        self.assertIn("strict_validation", detail)
        self.assertIn("legacy_validation", detail)

    def test_tracker_strict_mode_flags_version_mismatch(self) -> None:
        preprocessor = object()
        payload = {
            "preprocessor": preprocessor,
            "dropped_features": [],
            "pruned_features": {},
            "tracking_metadata": {
                "session_id": self.session_id,
                "problem_type": "regression",
                "feature_names_in": [],
                "library_versions": {
                    "python": "1.0.0",
                    "sklearn": "0.1.0",
                    "pandas": "0.1.0",
                    "numpy": "0.1.0",
                },
            },
        }

        result = validate_preprocessing_artifact(
            preprocessing_obj=payload,
            session_id=self.session_id,
            strict=True,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("sklearn version mismatch" in e for e in result["errors"]))

    def test_best_model_summary_rule_based_output(self) -> None:
        results = [
            {"Model": "Lasso", "R2_Score": 0.84, "RMSE": 1800.0},
            {"Model": "Ridge", "R2_Score": 0.83, "RMSE": 1850.0},
        ]
        summary = build_best_model_summary("regression", results)
        self.assertEqual(summary.get("best_model"), "Lasso")
        self.assertEqual(summary.get("best_metric"), "R2_Score")
        self.assertEqual(summary.get("source"), "rule-based")

    def test_best_model_summary_contains_gap_and_reason(self) -> None:
        results = [
            {"Model": "LogisticRegression", "Balanced_Accuracy": 0.86},
            {"Model": "DecisionTree", "Balanced_Accuracy": 0.81},
        ]
        summary = build_best_model_summary("classification", results)
        self.assertEqual(summary.get("best_model"), "LogisticRegression")
        self.assertEqual(summary.get("best_metric"), "Balanced_Accuracy")
        self.assertAlmostEqual(float(summary.get("score_gap_vs_runner_up", 0.0)), 0.05, places=3)
        self.assertTrue(isinstance(summary.get("reason"), str) and len(summary.get("reason")) > 0)

    def test_ml_models_request_rejects_invalid_cv(self) -> None:
        response = self.client.post(
            "/ml-models",
            json={
                "session_id": "abc",
                "problem_statement": "predict y",
                "cv": 1,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_preview_serializes_nan_and_inf(self) -> None:
        fake_df = pd.DataFrame(
            {
                "a": [1.0, np.nan],
                "b": [np.inf, -np.inf],
            }
        )

        class _FakeFeatureEngineer:
            def __init__(self, dataset):
                self.dataset = dataset

            def generate_features(self):
                return fake_df, "session_id_fake"

        csv_bytes = io.BytesIO(b"x,y\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n17,18\n19,20\n")
        files = {"file": ("sample.csv", csv_bytes, "text/csv")}

        with patch("app.main.FeatureEngineer1", _FakeFeatureEngineer):
            response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("preview", payload)
        self.assertTrue(isinstance(payload["preview"], list))


if __name__ == "__main__":
    unittest.main()
