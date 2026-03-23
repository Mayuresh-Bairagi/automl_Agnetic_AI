import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception
import joblib
from src.problem_statement.AutoFeatureSelector import FeatureSelector


class AutoMLClassifier:
    """Multi-algorithm classifier with automated hyperparameter tuning.

    Trains a suite of classification algorithms using GridSearchCV, evaluates
    each model on a held-out test set, and persists the best estimator along
    with all preprocessing artefacts so they can be reused for inference.

    Parameters
    ----------
    session_id : str
        Unique identifier for the current upload session; used to locate the
        processed dataset on disk.
    problem_statement : str
        Plain-language description of the ML task (e.g. "Predict churn").
    result : dict
        Output of :class:`~src.problem_statement.target_variable.TargetVariable`
        containing ``target_variable`` and ``problem_type``.
    df : pd.DataFrame
        Full processed DataFrame (used by the feature selector).
    test_size : float, optional
        Fraction of data reserved for testing (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        session_id: str,
        problem_statement: str,
        result: dict,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.logger = CustomLogger().get_logger(__file__)
        self.logger.info("Initializing AutoMLClassifier", session_id=session_id)

        try:
            self.database_path = os.path.join(
                os.getcwd(), "data", "datasetAnalysis", session_id, "processed_file.csv"
            )
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Dataset not found at {self.database_path}")

            self.df = pd.read_csv(self.database_path)
            if self.df.empty:
                raise ValueError(
                    "Processed dataset is empty after cleaning; cannot train classification models."
                )
            self.selector = FeatureSelector(session_id, problem_statement, result, df)
            self.context = self.selector.llm_response()

            if self.context is None or not isinstance(self.context, dict):
                self.logger.warning(
                    "Feature selector context missing or invalid; using fallback",
                    context_type=str(type(self.context)),
                )
                self.context = {
                    "target_col": result.get("target_variable") if isinstance(result, dict) else None,
                    "dropped_features": [],
                }

            self.target_col = self.context.get("target_col") or (result.get("target_variable") if isinstance(result, dict) else None)
            if not self.target_col:
                raise ValueError("Unable to determine target column from feature selector or target detection result")
            self.dropped_features: List[str] = self.context.get("dropped_features") or []
            self.test_size = test_size
            self.random_state = random_state
            self.output_dir = os.path.join(
                os.getcwd(), "data", "datasetAnalysis", session_id
            )
            os.makedirs(self.output_dir, exist_ok=True)

            self.label_encoders: Dict[str, LabelEncoder] = {}
            self.scaler: Optional[MinMaxScaler] = None
            self.best_model = None
            self.results_df: Optional[pd.DataFrame] = None
            self.trained_models: Dict = {}
            self.model_paths: Dict[str, str] = {}
            self.preprocessing_objects: Dict = {}

            self.logger.info("AutoMLClassifier initialized successfully")

        except Exception as e:
            self.logger.error("Initialization failed", error=str(e))
            raise AutoML_Exception("Initialization failed", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_label_transform(le: LabelEncoder, value: str) -> int:
        """Transform a single value using a fitted LabelEncoder.

        Returns ``-1`` for unseen labels so that test-set inference never
        crashes on out-of-vocabulary values.
        """
        return int(le.transform([value])[0]) if value in le.classes_ else -1

    def _encode_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode categorical columns; fit only on *X_train*."""
        for col in X_train.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].astype(str).map(
                lambda v, _le=le: self._safe_label_transform(_le, v)
            )
            self.label_encoders[col] = le
        return X_train, X_test

    def _scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """MinMax-scale numeric columns; fit only on *X_train*."""
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols):
            self.scaler = MinMaxScaler()
            X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        return X_train, X_test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Drop irrelevant features and encode the target label.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (categorical columns still as *object*).
        y : np.ndarray
            Encoded target vector.
        """
        try:
            self.logger.info("Starting preprocessing")

            df_cleaned = self.df.drop(columns=self.dropped_features, errors="ignore")
            X = df_cleaned.drop(self.target_col, axis=1)
            y = df_cleaned[self.target_col]

            # Encode target if categorical
            if y.dtype == "object" or str(y.dtype) == "category":
                target_le = LabelEncoder()
                y = target_le.fit_transform(y)
                self.label_encoders[self.target_col] = target_le

            self.logger.info(
                "Preprocessing completed",
                features=list(X.columns),
                target=self.target_col,
            )
            return X, y

        except Exception as e:
            self.logger.error("Preprocessing failed", error=str(e))
            raise AutoML_Exception("Preprocessing failed", e)

    def train_models(
        self, cv: int = 3, skip_heavy: bool = False
    ) -> Tuple[pd.DataFrame, Dict, Dict[str, str]]:
        """Train all classifiers with GridSearchCV and evaluate on the test set.

        Feature encoding and scaling are fit **exclusively on the training
        split** to prevent data leakage.

        Parameters
        ----------
        cv : int
            Number of cross-validation folds for GridSearchCV (default 3).
        skip_heavy : bool
            When ``True``, skip computationally expensive models (Random
            Forest, Gradient Boosting, SVC, XGBoost, LightGBM) to speed up
            iteration (default False).

        Returns
        -------
        results_df : pd.DataFrame
            Per-model evaluation metrics sorted by Accuracy (descending).
        trained_models : dict
            Mapping of model name → fitted best estimator.
        model_paths : dict
            Mapping of model name → path of the serialised ``.joblib`` file.
        """
        self.logger.info("Starting preprocessing inside train_models()")
        X, y = self.preprocess()

        # ---- Split FIRST to avoid any leakage ----
        try:
            self.logger.info("Splitting dataset", test_size=self.test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        except Exception as e:
            self.logger.error("Dataset split failed", error=str(e))
            raise AutoML_Exception("Dataset split failed", e)

        # ---- Fit encoders / scaler on train only ----
        X_train, X_test = self._encode_features(X_train.copy(), X_test.copy())
        X_train, X_test = self._scale_features(X_train, X_test)

        # Persist preprocessing artefacts
        self.preprocessing_objects = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
        }
        joblib.dump(
            self.preprocessing_objects,
            os.path.join(self.output_dir, "preprocessing.joblib"),
        )
        self.logger.info(
            "Preprocessing artefacts saved",
            numeric_features=list(
                X_train.select_dtypes(include=["int64", "float64"]).columns
            ),
            categorical_features=list(self.label_encoders.keys()),
        )

        # ---- Model catalogue ----
        models: Dict = {
            "LogisticRegression": (
                LogisticRegression(max_iter=3000, solver="saga"),
                {"C": [0.01, 0.1, 1, 10], "penalty": ["l2", "l1"]},
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=self.random_state),
                {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5]},
            ),
            "KNeighbors": (
                KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=self.random_state),
                {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
            "SVC": (
                SVC(probability=True),
                {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]},
            ),
        }

        if _XGBOOST_AVAILABLE:
            models["XGBoost"] = (
                XGBClassifier(
                    random_state=self.random_state,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    verbosity=0,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                },
            )

        if _LIGHTGBM_AVAILABLE:
            models["LightGBM"] = (
                LGBMClassifier(random_state=self.random_state, verbosity=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [-1, 6],
                    "learning_rate": [0.05, 0.1],
                    "num_leaves": [31, 63],
                },
            )

        if skip_heavy:
            heavy = {"RandomForest", "GradientBoosting", "SVC", "XGBoost", "LightGBM"}
            models = {k: v for k, v in models.items() if k not in heavy}

        n_classes = len(np.unique(y))
        avg_strategy = "binary" if n_classes == 2 else "weighted"
        roc_multi_class = "ovr"

        results = []
        best_accuracy = -float("inf")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, (model, params) in models.items():
            try:
                self.logger.info(f"Training {name}")
                gs = GridSearchCV(
                    model, params, cv=skf, n_jobs=-1, scoring="accuracy", refit=True
                )
                gs.fit(X_train, y_train)
                y_pred = gs.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                precision = precision_score(
                    y_test, y_pred, average=avg_strategy, zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average=avg_strategy, zero_division=0
                )

                # ROC-AUC — requires probability estimates
                try:
                    if hasattr(gs.best_estimator_, "predict_proba"):
                        y_proba = gs.predict_proba(X_test)
                        if n_classes == 2:
                            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:
                            roc_auc = roc_auc_score(
                                y_test,
                                y_proba,
                                multi_class=roc_multi_class,
                                average="macro",
                            )
                    else:
                        roc_auc = None
                except Exception:
                    roc_auc = None

                # Cross-val score on training set for an unbiased variance estimate
                cv_scores = cross_val_score(
                    gs.best_estimator_, X_train, y_train, cv=skf, scoring="accuracy"
                )

                row = {
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "F1_Score": round(f1, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else None,
                    "CV_Accuracy_Mean": round(cv_scores.mean(), 4),
                    "CV_Accuracy_Std": round(cv_scores.std(), 4),
                    "Best_Params": gs.best_params_,
                }
                results.append(row)

                model_path = os.path.join(self.output_dir, f"{name}.joblib")
                joblib.dump(gs.best_estimator_, model_path)
                self.trained_models[name] = gs.best_estimator_
                self.model_paths[name] = model_path

                if acc > best_accuracy:
                    best_accuracy = acc
                    self.best_model = gs.best_estimator_

                self.logger.info(
                    f"{name} trained",
                    Accuracy=acc,
                    F1_Score=f1,
                    Precision=precision,
                    Recall=recall,
                    ROC_AUC=roc_auc,
                    CV_Mean=cv_scores.mean(),
                    model_path=model_path,
                )

            except Exception as e:
                self.logger.error(f"{name} training failed", error=str(e))
                continue

        self.results_df = pd.DataFrame(results).sort_values(
            by="Accuracy", ascending=False
        ).reset_index(drop=True)
        self.logger.info(
            "All models training completed",
            trained_models=list(self.trained_models.keys()),
        )
        return self.results_df, self.trained_models, self.model_paths


if __name__ == "__main__":
    try:
        from src.problem_statement.target_variable import TargetVariable

        session_id = "session_id_20251103_194401_c64e583a"
        problem_statement = "Predict the RainToday weather condition"

        target_var_handler = TargetVariable(session_id=session_id)
        result, df = target_var_handler.get_target_variable(problem_statement)

        automl = AutoMLClassifier(session_id, problem_statement, result, df)
        results_df, trained_models, model_paths = automl.train_models(skip_heavy=True)

        print("Model Training Summary:")
        print(results_df)
        print("\nBest Model:", results_df.iloc[0]["Model"])
        print("Model Saved at:", model_paths[results_df.iloc[0]["Model"]])

    except Exception as e:
        print("Error during AutoML execution:", str(e))
