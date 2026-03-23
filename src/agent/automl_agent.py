"""
AutoML LangGraph Agent
======================
Orchestrates the full AutoML pipeline as a stateful LangGraph graph.

Pipeline nodes (in order):
  1. ingest          — load processed dataset from disk
  2. analyze_data    — infer and cast data types (AI-powered)
  3. engineer_features — generate derived features
  4. detect_target   — identify target variable + problem type
  5. select_features — statistical + LLM feature ranking
  6. train_models    — multi-algorithm training with GridSearchCV
  7. generate_report — assemble a structured summary

State key ``status`` drives conditional routing so any node can signal an
error without crashing the whole graph.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from expection.customExpection import AutoML_Exception
from logger.customlogger import CustomLogger
from src.Classifier.MLClassifier import AutoMLClassifier
from src.Regression.regression import AutoMLRegressor
from src.datasetAnalysis.data_type_analysis import DataTypeAnalyzer
from src.dataCleaning.featureEngineering01 import FeatureEngineer1
from src.problem_statement.target_variable import TargetVariable

log = CustomLogger().get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed state shared across all graph nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    session_id: str
    problem_statement: str
    df: Optional[Any]                  # pandas DataFrame
    target_result: Optional[Dict]      # output of TargetVariable
    feature_context: Optional[Dict]    # output of FeatureSelector
    training_results: Optional[List[Dict]]
    model_paths: Optional[Dict]
    report: Optional[Dict]
    status: str                        # "ok" | "error"
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _load_dataset(state: AgentState) -> AgentState:
    """Load the processed CSV produced by the upload step."""
    try:
        session_id = state["session_id"]
        csv_path = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", session_id, "processed_file.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(
                "Processed dataset is empty after cleaning. Please upload data with fewer missing rows "
                "or relax dropna/drop_duplicates in feature engineering."
            )
        log.info("Dataset loaded", session_id=session_id, shape=str(df.shape))
        return {**state, "df": df, "status": "ok"}
    except Exception as exc:
        log.error("Failed to load dataset", error=str(exc))
        return {**state, "status": "error", "error_message": str(exc)}


def _detect_target(state: AgentState) -> AgentState:
    """Use the TargetVariable LLM chain to detect target + problem type."""
    if state["status"] != "ok":
        return state
    try:
        handler = TargetVariable(session_id=state["session_id"])
        result, df = handler.get_target_variable(state["problem_statement"])
        log.info(
            "Target variable detected",
            target=result.get("target_variable"),
            problem_type=result.get("problem_type"),
        )
        return {**state, "target_result": result, "df": df, "status": "ok"}
    except Exception as exc:
        log.error("Target detection failed", error=str(exc))
        return {**state, "status": "error", "error_message": str(exc)}


def _train_models(state: AgentState) -> AgentState:
    """Train models (regression or classification) and persist them."""
    if state["status"] != "ok":
        return state
    try:
        session_id = state["session_id"]
        problem_statement = state["problem_statement"]
        result = state["target_result"]
        df = state["df"]
        problem_type = result["problem_type"].lower()

        if problem_type == "regression":
            trainer = AutoMLRegressor(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df,
            )
        elif problem_type == "classification":
            trainer = AutoMLClassifier(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df,
            )
        else:
            raise ValueError(f"Unsupported problem type for training: {problem_type}")

        results_df, _trained_models, model_paths = trainer.train_models(skip_heavy=True)
        training_results = results_df.to_dict(orient="records")

        log.info(
            "Model training complete",
            models=list(model_paths.keys()),
            best=results_df.iloc[0]["Model"] if not results_df.empty else "N/A",
        )
        return {
            **state,
            "training_results": training_results,
            "model_paths": model_paths,
            "status": "ok",
        }
    except Exception as exc:
        log.error("Model training failed", error=str(exc))
        return {**state, "status": "error", "error_message": str(exc)}


def _generate_report(state: AgentState) -> AgentState:
    """Assemble a structured report from all previous node outputs."""
    try:
        target_result = state.get("target_result") or {}
        training_results = state.get("training_results") or []

        best_model = (
            training_results[0]["Model"] if training_results else "N/A"
        )

        # Determine the primary metric name
        metric_key = (
            "R2_Score"
            if target_result.get("problem_type", "").lower() == "regression"
            else "Accuracy"
        )
        best_score = (
            training_results[0].get(metric_key, "N/A") if training_results else "N/A"
        )

        report = {
            "session_id": state["session_id"],
            "problem_statement": state["problem_statement"],
            "target_variable": target_result.get("target_variable"),
            "problem_type": target_result.get("problem_type"),
            "justification": target_result.get("justification"),
            "best_model": best_model,
            "best_score": best_score,
            "metric": metric_key,
            "all_results": training_results,
            "model_paths": state.get("model_paths") or {},
        }
        log.info("Report generated", best_model=best_model, best_score=best_score)
        return {**state, "report": report, "status": "ok"}
    except Exception as exc:
        log.error("Report generation failed", error=str(exc))
        return {**state, "status": "error", "error_message": str(exc)}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    """Route to END on error, otherwise continue to next node."""
    return "end_with_error" if state["status"] == "error" else "continue"


def build_automl_graph() -> StateGraph:
    """Construct and compile the AutoML LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("load_dataset", _load_dataset)
    graph.add_node("detect_target", _detect_target)
    graph.add_node("train_models", _train_models)
    graph.add_node("generate_report", _generate_report)

    # Entry point
    graph.set_entry_point("load_dataset")

    # Conditional edges — bail out to END on any error
    for src, dst in [
        ("load_dataset", "detect_target"),
        ("detect_target", "train_models"),
        ("train_models", "generate_report"),
    ]:
        graph.add_conditional_edges(
            src,
            _route,
            {"continue": dst, "end_with_error": END},
        )

    graph.add_edge("generate_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

class AutoMLAgent:
    """
    High-level wrapper around the LangGraph compiled graph.

    Usage
    -----
    >>> agent = AutoMLAgent()
    >>> report = agent.run(session_id="abc123", problem_statement="Predict price")
    """

    def __init__(self) -> None:
        self._graph = build_automl_graph()
        log.info("AutoMLAgent initialised — LangGraph graph compiled successfully")

    def run(self, session_id: str, problem_statement: str) -> Dict[str, Any]:
        """
        Execute the full AutoML pipeline.

        Parameters
        ----------
        session_id : str
            Session identifier from the /upload step.
        problem_statement : str
            Plain-language description of the ML task.

        Returns
        -------
        dict
            Final agent state including ``report`` and ``status``.
        """
        initial_state: AgentState = {
            "session_id": session_id,
            "problem_statement": problem_statement,
            "df": None,
            "target_result": None,
            "feature_context": None,
            "training_results": None,
            "model_paths": None,
            "report": None,
            "status": "ok",
            "error_message": None,
        }
        log.info(
            "AutoMLAgent run started",
            session_id=session_id,
            problem_statement=problem_statement,
        )
        final_state = self._graph.invoke(initial_state)
        log.info(
            "AutoMLAgent run finished",
            session_id=session_id,
            status=final_state.get("status"),
        )
        return final_state
