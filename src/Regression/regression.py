import os
import sys
import importlib
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVR

XGBRegressor = None
LGBMRegressor = None
_XGBOOST_AVAILABLE = False
_LIGHTGBM_AVAILABLE = False

try:
    XGBRegressor = importlib.import_module("xgboost").XGBRegressor
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False

try:
    LGBMRegressor = importlib.import_module("lightgbm").LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except Exception:
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
        result: Optional[dict] = None,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_llm_feature_selection: bool = True,
    ) -> None:
        self.logger = CustomLogger().get_logger(__file__)
        self.logger.info("Initializing AutoMLRegressor", session_id=session_id)

        try:
            self.session_id = session_id
            self.problem_statement = problem_statement
            self.result = result if isinstance(result, dict) else {}
            self.database_path = os.path.join(
                os.getcwd(), "data", "datasetAnalysis", session_id, "processed_file.csv"
            )
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Dataset not found at {self.database_path}")

            if isinstance(df, pd.DataFrame) and not df.empty:
                self.df = df.copy()
            else:
                self.df = pd.read_csv(self.database_path)
            if self.df.empty:
                raise ValueError(
                    "Processed dataset is empty after cleaning; cannot train regression models."
                )

            self.target_col = self.result.get("target_variable")
            if not self.target_col:
                raise ValueError("Unable to determine target column from target detection result")
            if self.target_col not in self.df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in processed dataset")

            self.dropped_features: List[str] = []
            self.test_size = test_size
            self.random_state = random_state
            self.use_llm_feature_selection = use_llm_feature_selection
            self.output_dir = os.path.join(
                os.getcwd(), "data", "datasetAnalysis", session_id
            )
            os.makedirs(self.output_dir, exist_ok=True)

            self.preprocessor: Optional[ColumnTransformer] = None
            self.best_model = None
            self.results_df: Optional[pd.DataFrame] = None
            self.trained_models: Dict = {}
            self.model_paths: Dict[str, str] = {}
            self.preprocessing_objects: Dict = {}

            self.logger.info("AutoMLRegressor initialized successfully")

        except Exception as e:
            self.logger.error("Initialization failed", error=str(e))
            raise AutoML_Exception("Initialization failed", cast(Any, sys))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_preprocessor(self, X_train: pd.DataFrame) -> Tuple[Any, List[str], List[str]]:
        """Fit preprocessing pipeline on train only (impute + encode + scale)."""
        categorical_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )
        X_train_t = self.preprocessor.fit_transform(X_train)
        return X_train_t, numeric_cols, categorical_cols

    def _transform_with_preprocessor(self, X_test: pd.DataFrame) -> Any:
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted")
        return self.preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Build X, y ready for splitting.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (categorical columns still as *object*).
        y : pd.Series
            Numeric target vector.
        """
        try:
            self.logger.info("Starting preprocessing")

            before_rows = len(self.df)
            df_work = self.df.drop_duplicates().copy()
            removed_duplicates = before_rows - len(df_work)
            if removed_duplicates > 0:
                self.logger.warning("Removed duplicate rows before split", removed_duplicates=removed_duplicates)

            df_work = df_work[df_work[self.target_col].notna()].copy()
            if df_work.empty:
                raise ValueError("No rows remaining after dropping missing target values")

            X = df_work.drop(self.target_col, axis=1)
            y = df_work[self.target_col]

            if self.target_col in X.columns:
                raise ValueError("Target leakage detected: target column present in features")

            if X.empty:
                raise ValueError("Feature matrix is empty after preprocessing")

            self.logger.info(
                "Preprocessing completed",
                features=list(X.columns),
                target=self.target_col,
            )
            return X, y

        except Exception as e:
            self.logger.error("Preprocessing failed", error=str(e))
            raise AutoML_Exception("Preprocessing failed", cast(Any, sys))

    def _resolve_train_only_dropped_features(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[str]:
        """Run feature selection strictly on training data to avoid leakage."""
        if not self.use_llm_feature_selection:
            self.logger.info("LLM feature selection disabled; using deterministic no-drop fallback")
            return []

        try:
            train_df = X_train.copy()
            train_df[self.target_col] = y_train

            selector = FeatureSelector(
                self.session_id,
                self.problem_statement,
                self.result,
                train_df,
            )
            context = selector.llm_response()

            if not isinstance(context, dict):
                self.logger.warning(
                    "Feature selector context invalid after split; using no dropped features",
                    context_type=str(type(context)),
                )
                return []

            dropped = context.get("dropped_features") or []
            if not isinstance(dropped, list):
                self.logger.warning(
                    "Dropped features not a list; ignoring",
                    dropped_type=str(type(dropped)),
                )
                return []

            dropped = [c for c in dropped if c != self.target_col]
            return dropped
        except Exception as e:
            self.logger.warning(
                "Feature selection failed after split; proceeding without dropped features",
                error=str(e),
            )
            return []

    @staticmethod
    def _resolve_train_only_pruned_features(X_train: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify noisy features using train split only to avoid leakage."""
        if X_train is None or X_train.empty:
            return {"high_missing": [], "constant": [], "high_cardinality": []}

        row_count = max(1, len(X_train))
        high_missing = [
            c for c in X_train.columns
            if float(X_train[c].isna().mean()) > 0.6
        ]
        constant = [
            c for c in X_train.columns
            if int(X_train[c].nunique(dropna=False)) <= 1
        ]
        high_cardinality = []
        for c in X_train.select_dtypes(include=["object", "category", "string"]).columns:
            unique_count = int(X_train[c].nunique(dropna=True))
            if unique_count > 100 and (unique_count / row_count) > 0.5:
                high_cardinality.append(c)

        return {
            "high_missing": sorted(set(high_missing)),
            "constant": sorted(set(constant)),
            "high_cardinality": sorted(set(high_cardinality)),
        }

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
            raise AutoML_Exception("Dataset split failed", cast(Any, sys))

        # Resolve dropped features on train split only (leakage-safe).
        self.dropped_features = self._resolve_train_only_dropped_features(X_train, y_train)
        if self.dropped_features:
            valid_dropped = [c for c in self.dropped_features if c in X_train.columns]
            X_train = X_train.drop(columns=valid_dropped, errors="ignore")
            X_test = X_test.drop(columns=valid_dropped, errors="ignore")
            self.logger.info("Applied train-only dropped features", dropped_features=valid_dropped)

        # Train-only pruning for noisy columns.
        pruned_map = self._resolve_train_only_pruned_features(X_train)
        prune_candidates = sorted(
            set(pruned_map["high_missing"]) | set(pruned_map["constant"]) | set(pruned_map["high_cardinality"])
        )
        if prune_candidates:
            X_train = X_train.drop(columns=prune_candidates, errors="ignore")
            X_test = X_test.drop(columns=prune_candidates, errors="ignore")
            self.logger.info("Applied train-only noisy-feature pruning", pruned_features=pruned_map)

        if X_train.shape[1] == 0:
            raise ValueError("All features were pruned/dropped; cannot train model")

        # Detect accidental train/test overlap by row fingerprint.
        train_hash = pd.util.hash_pandas_object(X_train, index=False).astype(str)
        test_hash = pd.util.hash_pandas_object(X_test, index=False).astype(str)
        train_fingerprints = set(train_hash.tolist())
        overlap_mask = test_hash.isin(train_fingerprints)
        overlap_count = int(overlap_mask.sum())
        if overlap_count > 0:
            self.logger.warning(
                "Potential duplicate leakage across train/test split; removing overlaps from test set",
                overlap_rows=overlap_count,
            )
            keep_mask = ~overlap_mask
            X_test = X_test.loc[keep_mask].copy()
            y_test = y_test.loc[keep_mask].copy() if hasattr(y_test, "loc") else np.asarray(y_test)[keep_mask.to_numpy()]
            if len(X_test) == 0:
                raise ValueError("All test rows overlapped with training rows after dedupe enforcement")

        # Fit preprocessing pipeline on train only
        X_train_t, numeric_cols, categorical_cols = self._fit_preprocessor(X_train.copy())
        X_test_t = self._transform_with_preprocessor(X_test.copy())

        # Persist preprocessing artefacts
        self.preprocessing_objects = {
            "preprocessor": self.preprocessor,
            "dropped_features": self.dropped_features,
            "pruned_features": pruned_map,
        }
        joblib.dump(
            self.preprocessing_objects,
            os.path.join(self.output_dir, "preprocessing.joblib"),
        )
        self.logger.info(
            "Preprocessing artefacts saved",
            numeric_features=numeric_cols,
            categorical_features=categorical_cols,
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

        if _XGBOOST_AVAILABLE and XGBRegressor is not None:
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

        if _LIGHTGBM_AVAILABLE and LGBMRegressor is not None:
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
                gs.fit(X_train_t, y_train)
                y_pred = gs.predict(X_test_t)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                # Cross-val score on training set
                cv_scores = cross_val_score(
                    gs.best_estimator_, X_train_t, y_train, cv=kf, scoring="r2"
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

        results_df = pd.DataFrame(results).sort_values(
            by="R2_Score", ascending=False
        ).reset_index(drop=True)
        if results_df.empty:
            raise ValueError("No regression models were successfully trained")
        self.results_df = results_df
        self.logger.info(
            "All models training completed",
            trained_models=list(self.trained_models.keys()),
        )
        return results_df, self.trained_models, self.model_paths


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
