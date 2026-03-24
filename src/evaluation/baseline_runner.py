import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Allow running this file directly: `python src/evaluation/baseline_runner.py ...`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from expection.customExpection import AutoML_Exception
from src.Classifier.MLClassifier import AutoMLClassifier
from src.Regression.regression import AutoMLRegressor
from src.problem_statement.target_variable import TargetVariable


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types for JSON serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _load_processed_df(session_id: str) -> pd.DataFrame:
    processed_path = os.path.join(
        os.getcwd(), "data", "datasetAnalysis", session_id, "processed_file.csv"
    )
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")
    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    return df


def _resolve_target(
    session_id: str,
    df: pd.DataFrame,
    problem_statement: Optional[str],
    target_col: Optional[str],
    problem_type: Optional[str],
) -> Dict[str, Any]:
    if target_col and problem_type:
        if target_col not in df.columns:
            raise ValueError(f"Provided target_col '{target_col}' not found in dataset.")
        return {
            "target_variable": target_col,
            "problem_type": problem_type.lower(),
            "justification": "Provided via CLI arguments",
        }

    if not problem_statement:
        raise ValueError(
            "Either provide both --target-col and --problem-type, or provide --problem-statement."
        )

    target_var_handler = TargetVariable(session_id=session_id)
    result, _ = target_var_handler.get_target_variable(problem_statement)

    if result.get("target_variable") not in df.columns:
        # Deterministic fallback: use last column and infer task type.
        fallback_target = str(df.columns[-1])
        target_series = df[fallback_target]
        numeric_target = pd.to_numeric(target_series, errors="coerce")
        numeric_ratio = float(numeric_target.notna().mean()) if len(target_series) else 0.0
        unique_ratio = float(target_series.nunique(dropna=True) / max(1, len(target_series)))

        inferred_problem_type = (
            "regression" if numeric_ratio >= 0.9 and unique_ratio > 0.05 else "classification"
        )

        return {
            "target_variable": fallback_target,
            "problem_type": inferred_problem_type,
            "justification": "Fallback target used because LLM target was invalid",
        }

    return result


def _dataset_quality_snapshot(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    top_missing = {
        str(col): round(float(pct), 4)
        for col, pct in missing_pct[missing_pct > 0].head(15).items()
    }

    duplicate_rows = int(df.duplicated().sum())

    target_distribution = None
    if target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False)
        target_distribution = {
            str(k): int(v) for k, v in counts.head(30).to_dict().items()
        }

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "top_missing_percent": top_missing,
        "target_distribution": target_distribution,
    }


def run_baseline(
    session_id: str,
    problem_statement: Optional[str],
    target_col: Optional[str],
    problem_type: Optional[str],
    skip_heavy: bool,
    cv: int,
    max_rows: Optional[int],
) -> Dict[str, Any]:
    df = _load_processed_df(session_id)

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        # Deterministic sampling for reproducible baseline timing.
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    result = _resolve_target(
        session_id=session_id,
        df=df,
        problem_statement=problem_statement,
        target_col=target_col,
        problem_type=problem_type,
    )

    detected_problem_type = str(result["problem_type"]).lower()
    quality = _dataset_quality_snapshot(df, result["target_variable"])

    if detected_problem_type == "classification":
        trainer = AutoMLClassifier(
            session_id=session_id,
            problem_statement=problem_statement or "",
            result=result,
            df=df,
            use_llm_feature_selection=False,
        )
    elif detected_problem_type == "regression":
        trainer = AutoMLRegressor(
            session_id=session_id,
            problem_statement=problem_statement or "",
            result=result,
            df=df,
            use_llm_feature_selection=False,
        )
    else:
        raise ValueError(f"Unsupported problem type for baseline run: {detected_problem_type}")

    results_df, _, model_paths = trainer.train_models(cv=cv, skip_heavy=skip_heavy)
    if results_df.empty:
        raise ValueError("No model results were produced during baseline run.")

    metric = "Accuracy" if detected_problem_type == "classification" else "R2_Score"
    best_row = results_df.iloc[0].to_dict()

    baseline_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "problem_statement": problem_statement,
        "problem_type": detected_problem_type,
        "target_variable": result.get("target_variable"),
        "dataset_quality": quality,
        "metric": metric,
        "best_model": best_row.get("Model"),
        "best_score": best_row.get(metric),
        "best_row": _json_safe(best_row),
        "all_results": _json_safe(results_df.to_dict(orient="records")),
        "model_paths": _json_safe(model_paths),
    }

    output_path = os.path.join(
        os.getcwd(), "data", "datasetAnalysis", session_id, "baseline_metrics.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(baseline_report), f, indent=2)

    baseline_report["output_path"] = output_path
    return baseline_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible baseline on the current AutoML pipeline and save "
            "quality + model metrics to baseline_metrics.json"
        )
    )
    parser.add_argument("--session-id", required=True, help="Session ID under data/datasetAnalysis")
    parser.add_argument(
        "--problem-statement",
        default=None,
        help="Natural language problem statement (used for target detection if target is not provided)",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional explicit target column name",
    )
    parser.add_argument(
        "--problem-type",
        default=None,
        choices=["classification", "regression"],
        help="Optional explicit problem type (required when --target-col is provided)",
    )
    parser.add_argument(
        "--no-skip-heavy",
        action="store_true",
        help="Run heavy models as well (default skips heavy models for faster baseline).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=2,
        help="Cross-validation folds for GridSearchCV (default 2 for faster baseline runs).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=3000,
        help="Deterministic row cap for quick baseline runs (default 3000). Use 0 for full dataset.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        report = run_baseline(
            session_id=args.session_id,
            problem_statement=args.problem_statement,
            target_col=args.target_col,
            problem_type=args.problem_type,
            skip_heavy=not args.no_skip_heavy,
            cv=max(2, int(args.cv)),
            max_rows=None if int(args.max_rows) <= 0 else int(args.max_rows),
        )
        print(json.dumps(_json_safe(report), indent=2))
    except (AutoML_Exception, Exception) as exc:
        raise SystemExit(f"Baseline run failed: {exc}")


if __name__ == "__main__":
    main()
