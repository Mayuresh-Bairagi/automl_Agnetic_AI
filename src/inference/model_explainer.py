from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pick_primary_metric(problem_type: str, row: Dict[str, Any]) -> Tuple[str, float, bool]:
    p = str(problem_type).lower()
    if p == "classification":
        for key in ("Balanced_Accuracy", "Accuracy", "F1_Score", "ROC_AUC"):
            val = _safe_float(row.get(key))
            if val is not None:
                return key, val, True
    else:
        for key in ("R2_Score", "CV_R2_Mean"):
            val = _safe_float(row.get(key))
            if val is not None:
                return key, val, True
        for key in ("RMSE", "MAE"):
            val = _safe_float(row.get(key))
            if val is not None:
                return key, val, False
    return "score", 0.0, True


def _rule_based_summary(problem_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "best_model": None,
            "reason": "No model results available to explain.",
            "source": "rule-based",
        }

    best = results[0]
    second = results[1] if len(results) > 1 else None

    metric, best_score, higher_better = _pick_primary_metric(problem_type, best)
    second_score = _safe_float(second.get(metric)) if second else None

    gap = None
    if second_score is not None:
        gap = best_score - second_score if higher_better else second_score - best_score

    best_model = str(best.get("Model", "Unknown"))
    runner_up = str(second.get("Model", "")) if second else None

    if gap is None:
        reason = (
            f"{best_model} ranked highest by {metric} ({best_score:.4f}) among available models."
        )
    else:
        reason = (
            f"{best_model} ranked highest by {metric} ({best_score:.4f}) and outperformed "
            f"{runner_up} by {gap:.4f} on the same metric."
        )

    return {
        "best_model": best_model,
        "best_metric": metric,
        "best_score": best_score,
        "runner_up_model": runner_up,
        "score_gap_vs_runner_up": gap,
        "reason": reason,
        "source": "rule-based",
    }


def build_best_model_summary(problem_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _rule_based_summary(problem_type=problem_type, results=results)

    use_llm = str(os.getenv("MODEL_EXPLANATION_USE_LLM", "false")).lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_llm or not results:
        return summary

    try:
        from utils.model_loader import ModelLoader

        llm = ModelLoader().load_llm()
        top_rows = results[:3]
        prompt = (
            "You are explaining model selection to a non-technical user. "
            "Given problem type and top model rows, explain in 2 concise lines why best model was selected. "
            "Be metric-focused and clear.\n"
            f"Problem type: {problem_type}\n"
            f"Top rows: {top_rows}\n"
            "Return plain text only."
        )
        llm_resp = llm.invoke(prompt)
        llm_content = getattr(llm_resp, "content", "")
        if isinstance(llm_content, str):
            llm_reason = llm_content.strip()
        else:
            llm_reason = str(llm_content).strip()

        if llm_reason:
            summary["reason"] = llm_reason
            summary["source"] = "llm"
    except Exception as exc:
        summary["llm_fallback_error"] = str(exc)

    return summary
