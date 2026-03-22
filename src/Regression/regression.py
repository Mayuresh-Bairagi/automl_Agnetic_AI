import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception
from src.problem_statement.AutoFeatureSelector import FeatureSelector


class AutoMLRegressor:
    """Multi-algorithm regressor with automated hyperparameter tuning.

    Trains a suite of regression algorithms using GridSearchCV, evaluates
    each model on a held-out test set (R², MAE, RMSE), and persists the best
    estimator together with all preprocessing artefacts.

    Parameters
    ----------
    session_id : str
        Unique identifier for the current upload session; used to locate the
        processed dataset on disk.
    problem_statement : str
        Plain-language description of the ML task (e.g. "Predict house price").
    result : dict
        Output of TargetVariable containing ``target_variable`` and ``problem_type``.
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
        result: dict = None,
        df: pd.DataFrame = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.logger = CustomLogger().get_logger(__file__)
        self.logger.info("Initializing AutoMLRegressor", session_id=session_id)

        try:
            self.database_path = os.path.join(
                os.getcwd(), "data", "datasetAnalysis", session_id, "processed_file.csv"
            )
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Dataset not found at {self.database_path}")

            self.df = pd.read_csv(self.database_path)
            self.selector = FeatureSelector(session_id, problem_statement, result, df)
            self.context = self.selector.llm_response()

            self.target_col = self.context["target_col"]
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

            self.logger.info("AutoMLRegressor initialized successfully")

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
        """Drop irrelevant features and return X, y ready for splitting.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (categorical columns still as *object*).
        y : pd.Series
            Numeric target vector.
        """
        try:
            self.logger.info("Starting preprocessing")

            df_cleaned = self.df.drop(columns=self.dropped_features, errors="ignore")
            X = df_cleaned.drop(self.target_col, axis=1)
            y = df_cleaned[self.target_col]

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
        """Train all regressors with GridSearchCV and evaluate on the test set.

        Feature encoding and scaling are fit **exclusively on the training
        split** to prevent data leakage.

        Parameters
        ----------
        cv : int
            Number of cross-validation folds for GridSearchCV (default 3).
        skip_heavy : bool
            When ``True``, skip computationally expensive models (Random
            Forest, Gradient Boosting, SVR, XGBoost, LightGBM) to speed up
            iteration (default False).

        Returns
        -------
        results_df : pd.DataFrame
            Per-model metrics (R2, MAE, RMSE) sorted by R2 (descending).
        trained_models : dict
            Mapping of model name to fitted best estimator.
        model_paths : dict
            Mapping of model name to path of the serialised .joblib file.
        """
        self.logger.info("Starting preprocessing inside train_models()")
        X, y = self.preprocess()

        # Split FIRST to avoid any leakage
        try:
            self.logger.info("Splitting dataset", test_size=self.test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        except Exception as e:
            self.logger.error("Dataset split failed", error=str(e))
            raise AutoML_Exception("Dataset split failed", e)

        # Fit encoders / scaler on train only
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
            categorical_features=list(self.label_encoders.keys()),
        )

        # Model catalogue
        models: Dict = {
            "LinearRegression": (
                LinearRegression(),
                {"fit_intercept": [True, False]},
            ),
            "Ridge": (
                Ridge(),
                {"alpha": [0.01, 0.1, 1, 10, 100]},
            ),
            "Lasso": (
                Lasso(max_iter=5000),
                {"alpha": [0.001, 0.01, 0.1, 1]},
            ),
            "ElasticNet": (
                ElasticNet(max_iter=5000),
                {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},
            ),
            "RandomForest": (
                RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(random_state=self.random_state),
                {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5],
                },
            ),
            "SVR": (
                SVR(),
                {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "epsilon": [0.01, 0.1]},
            ),
        }

        if _XGBOOST_AVAILABLE:
            models["XGBoost"] = (
                XGBRegressor(
                    random_state=self.random_state,
                    verbosity=0,
                    objective="reg:squarederror",
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
                LGBMRegressor(random_state=self.random_state, verbosity=-1),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [-1, 6],
                    "learning_rate": [0.05, 0.1],
                    "num_leaves": [31, 63],
                },
            )

        if skip_heavy:
            heavy = {"RandomForest", "GradientBoosting", "SVR", "XGBoost", "LightGBM"}
            models = {k: v for k, v in models.items() if k not in heavy}

        results = []
        best_r2 = -float("inf")
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, (model, params) in models.items():
            try:
                self.logger.info(f"Training {name}")
                gs = GridSearchCV(
                    model, params, cv=kf, n_jobs=-1, scoring="r2", refit=True
                )
                gs.fit(X_train, y_train)
                y_pred = gs.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                # Cross-val score on training set
                cv_scores = cross_val_score(
                    gs.best_estimator_, X_train, y_train, cv=kf, scoring="r2"
                )

                row = {
                    "Model": name,
                    "R2_Score": round(r2, 4),
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "CV_R2_Mean": round(cv_scores.mean(), 4),
                    "CV_R2_Std": round(cv_scores.std(), 4),
                    "Best_Params": gs.best_params_,
                }
                results.append(row)

                model_path = os.path.join(self.output_dir, f"{name}.joblib")
                joblib.dump(gs.best_estimator_, model_path)
                self.trained_models[name] = gs.best_estimator_
                self.model_paths[name] = model_path

                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model = gs.best_estimator_

                self.logger.info(
                    f"{name} trained",
                    R2_Score=r2,
                    MAE=mae,
                    RMSE=rmse,
                    CV_Mean=cv_scores.mean(),
                    model_path=model_path,
                )

            except Exception as e:
                self.logger.error(f"{name} training failed", error=str(e))
                continue

        self.results_df = pd.DataFrame(results).sort_values(
            by="R2_Score", ascending=False
        ).reset_index(drop=True)
        self.logger.info(
            "All models training completed",
            trained_models=list(self.trained_models.keys()),
        )
        return self.results_df, self.trained_models, self.model_paths


if __name__ == "__main__":
    try:
        from src.problem_statement.target_variable import TargetVariable

        session_id = "session_id_20251004_193459_4af2e06b"
        problem_statement = "Predict the ticket price"

        target_var_handler = TargetVariable(session_id=session_id)
        result, df = target_var_handler.get_target_variable(problem_statement)

        automl = AutoMLRegressor(
            session_id, problem_statement, result=result, df=df
        )
        results_df, trained_models, model_paths = automl.train_models(skip_heavy=True)

        print("Model Training Summary")
        print(results_df)
        print("\nBest Model:", results_df.iloc[0]["Model"])
        print("Model Saved at:", model_paths[results_df.iloc[0]["Model"]])

    except Exception as e:
        print("Error during AutoML execution:", str(e))
