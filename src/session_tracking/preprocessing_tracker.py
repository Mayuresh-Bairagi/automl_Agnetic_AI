from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List
import sys

import joblib


_REQUIRED_KEYS = {"preprocessor", "dropped_features", "pruned_features"}


def _runtime_versions() -> Dict[str, str]:
    def _pkg_version(name: str) -> str:
        try:
            mod = importlib.import_module(name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "sklearn": _pkg_version("sklearn"),
        "pandas": _pkg_version("pandas"),
        "numpy": _pkg_version("numpy"),
    }


def _major_minor(version_text: str) -> str:
    try:
        parts = str(version_text).split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else str(version_text)
    except Exception:
        return str(version_text)


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def validate_preprocessing_artifact(
    preprocessing_obj: Dict[str, Any],
    session_id: str,
    strict: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(preprocessing_obj, dict):
        return {
            "valid": False,
            "errors": ["preprocessing artifact is not a dictionary"],
            "warnings": [],
            "metadata": {},
        }

    missing_keys = sorted(list(_REQUIRED_KEYS - set(preprocessing_obj.keys())))
    if missing_keys:
        if strict:
            errors.append(f"Missing required preprocessing keys: {missing_keys}")
        else:
            warnings.append(f"Missing recommended preprocessing keys: {missing_keys}")

    dropped_features = preprocessing_obj.get("dropped_features")
    if not isinstance(dropped_features, list):
        if strict:
            errors.append("dropped_features must be a list")
        else:
            warnings.append("dropped_features is not a list; treating as empty")

    pruned_features = preprocessing_obj.get("pruned_features")
    if not isinstance(pruned_features, dict):
        if strict:
            errors.append("pruned_features must be a dictionary")
        else:
            warnings.append("pruned_features is not a dictionary; treating as empty")

    tracking_metadata = preprocessing_obj.get("tracking_metadata")
    if not isinstance(tracking_metadata, dict):
        if strict:
            errors.append("tracking_metadata missing in strict mode")
        else:
            warnings.append("tracking_metadata missing; training lineage visibility is reduced")
        tracking_metadata = {}

    tracked_session_id = tracking_metadata.get("session_id")
    if tracked_session_id and tracked_session_id != session_id:
        errors.append(
            f"tracking_metadata.session_id mismatch: expected {session_id}, got {tracked_session_id}"
        )

    tracked_problem_type = tracking_metadata.get("problem_type")
    if tracked_problem_type and str(tracked_problem_type).lower() not in {"regression", "classification"}:
        warnings.append("tracking_metadata.problem_type has unexpected value")

    feature_names_in = tracking_metadata.get("feature_names_in")
    if feature_names_in is not None and not isinstance(feature_names_in, list):
        warnings.append("tracking_metadata.feature_names_in should be a list")

    current_versions = _runtime_versions()
    tracked_versions = tracking_metadata.get("library_versions")
    if isinstance(tracked_versions, dict):
        tracked_sklearn = str(tracked_versions.get("sklearn", "unknown"))
        current_sklearn = str(current_versions.get("sklearn", "unknown"))

        if tracked_sklearn != "unknown" and current_sklearn != "unknown":
            if _major_minor(tracked_sklearn) != _major_minor(current_sklearn):
                msg = (
                    f"sklearn version mismatch: trained={tracked_sklearn}, runtime={current_sklearn}"
                )
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        for pkg in ("pandas", "numpy", "python"):
            trained_v = str(tracked_versions.get(pkg, "unknown"))
            runtime_v = str(current_versions.get(pkg, "unknown"))
            if trained_v != "unknown" and runtime_v != "unknown" and _major_minor(trained_v) != _major_minor(runtime_v):
                warnings.append(
                    f"{pkg} version mismatch: trained={trained_v}, runtime={runtime_v}"
                )
    else:
        if strict:
            errors.append("tracking_metadata.library_versions missing in strict mode")
        else:
            warnings.append("tracking_metadata.library_versions missing")

    preprocessor = preprocessing_obj.get("preprocessor")
    if preprocessor is None:
        errors.append("preprocessor is missing")
    else:
        inferred_feature_names = list(getattr(preprocessor, "feature_names_in_", []))
        if not inferred_feature_names:
            warnings.append(
                "preprocessor has no feature_names_in_; schema compatibility checks will be weaker"
            )
        if isinstance(feature_names_in, list) and inferred_feature_names and feature_names_in != inferred_feature_names:
            errors.append(
                "tracked feature_names_in does not match preprocessor.feature_names_in_"
            )

    cleaner = preprocessing_obj.get("cleaner")
    if cleaner is None:
        warnings.append("cleaner missing; robust text/schema normalization may be reduced")
    else:
        if not hasattr(cleaner, "transform"):
            msg = "cleaner object does not implement transform"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)

    cleaning_audit = tracking_metadata.get("cleaning_audit")
    if cleaning_audit is not None and not isinstance(cleaning_audit, dict):
        warnings.append("tracking_metadata.cleaning_audit should be a dictionary")

    if not _as_list(preprocessing_obj.get("dropped_features")) and not _as_list(
        tracking_metadata.get("dropped_features")
    ):
        warnings.append("No dropped_features recorded in artifact or tracking metadata")

    return {
        "valid": len(errors) == 0,
        "strict": bool(strict),
        "errors": errors,
        "warnings": warnings,
        "metadata": tracking_metadata,
        "runtime_versions": current_versions,
    }


def load_and_validate_preprocessing_artifact(
    session_path: Path,
    session_id: str,
    strict: bool = False,
) -> Dict[str, Any]:
    preprocessing_path = session_path / "preprocessing.joblib"

    if not preprocessing_path.exists():
        return {
            "valid": False,
            "strict": bool(strict),
            "errors": ["preprocessing.joblib not found"],
            "warnings": [],
            "metadata": {},
            "runtime_versions": _runtime_versions(),
        }

    try:
        preprocessing_obj = joblib.load(preprocessing_path)
    except Exception as exc:
        return {
            "valid": False,
            "strict": bool(strict),
            "errors": [f"Failed to load preprocessing artifact: {str(exc)}"],
            "warnings": [],
            "metadata": {},
            "runtime_versions": _runtime_versions(),
        }

    return validate_preprocessing_artifact(
        preprocessing_obj,
        session_id=session_id,
        strict=strict,
    )
