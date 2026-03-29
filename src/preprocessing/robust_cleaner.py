from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


_PLACEHOLDER_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "nil",
    "nan",
    "-",
    "--",
    "?",
    "unknown",
    "not available",
    "not_applicable",
}

_TRUE_VALUES = {"true", "1", "yes", "y", "t", "on"}
_FALSE_VALUES = {"false", "0", "no", "n", "f", "off"}

_CURRENCY_RE = re.compile(r"[\$€£₹¥]")
_MULTISPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_INVALID_COL_RE = re.compile(r"[^0-9a-zA-Z_]+")

_COMMON_CATEGORY_ALIASES = {
    "ny": "new york",
    "newyork": "new york",
    "la": "los angeles",
    "sf": "san francisco",
    "u.s.": "united states",
    "us": "united states",
    "usa": "united states",
    "u.k.": "united kingdom",
    "uk": "united kingdom",
}


def _normalize_column_name_raw(name: Any) -> str:
    text = unicodedata.normalize("NFKC", str(name))
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.strip().lower()
    text = _MULTISPACE_RE.sub("_", text)
    text = _INVALID_COL_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "column"


def _jsonify_nested(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    return value


def _decode_bytes(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1", errors="ignore")


def _normalize_text_scalar(value: Any) -> Any:
    if value is None:
        return np.nan

    if isinstance(value, bytes):
        value = _decode_bytes(value)

    value = _jsonify_nested(value)

    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return np.nan
        return value

    text = unicodedata.normalize("NFKC", str(value))
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("�", "")
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if not text:
        return np.nan

    lower = text.lower()
    if lower in _PLACEHOLDER_VALUES:
        return np.nan

    # Canonicalize embedded json-like strings when possible.
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except Exception:
            pass

    return text


def _normalize_text_series(series: pd.Series, lowercase: bool = True) -> pd.Series:
    cleaned = series.map(_normalize_text_scalar)
    cleaned = cleaned.replace({"": np.nan})
    if lowercase:
        cleaned = cleaned.map(lambda v: v.lower() if isinstance(v, str) else v)
    return cleaned


def _coerce_boolean_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_text_series(series, lowercase=True)

    def _map_bool(v: Any) -> Any:
        if pd.isna(v):
            return np.nan
        if isinstance(v, bool):
            return float(v)
        sv = str(v).strip().lower()
        if sv in _TRUE_VALUES:
            return 1.0
        if sv in _FALSE_VALUES:
            return 0.0
        return np.nan

    return cleaned.map(_map_bool)


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        return num.replace([np.inf, -np.inf], np.nan)

    cleaned = _normalize_text_series(series, lowercase=True).astype("string")
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(_CURRENCY_RE, "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    cleaned = cleaned.str.replace(r"(?i)(kg|g|lbs?|lb|km|mi|mile|miles|cm|mm|m|in|ft|usd|eur|inr)$", "", regex=True)
    cleaned = cleaned.replace({"": np.nan, "-": np.nan, ".": np.nan, "-.": np.nan})
    num = pd.to_numeric(cleaned, errors="coerce")
    return num.replace([np.inf, -np.inf], np.nan)


def _coerce_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        return dt

    cleaned = _normalize_text_series(series, lowercase=False)
    try:
        dt = pd.to_datetime(cleaned, errors="coerce", utc=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(cleaned, errors="coerce", utc=True)
    return dt


class RobustTabularCleaner(BaseEstimator, TransformerMixin):
    """Robust dataframe cleaner fit on training data only.

    Performs strict text normalization, placeholder unification, type coercion,
    outlier clipping, skew handling, rare-category consolidation, datetime
    expansion, schema alignment, and sanity checks.
    """

    def __init__(
        self,
        problem_type: str,
        numeric_ratio_threshold: float = 0.9,
        datetime_ratio_threshold: float = 0.9,
        bool_ratio_threshold: float = 0.98,
        rare_freq_threshold: float = 0.01,
        rare_min_count: int = 20,
        high_cardinality_threshold: int = 250,
        high_cardinality_ratio: float = 0.7,
        clip_quantiles: Tuple[float, float] = (0.005, 0.995),
        max_text_length: int = 5000,
    ) -> None:
        self.problem_type = str(problem_type).lower()
        self.numeric_ratio_threshold = float(numeric_ratio_threshold)
        self.datetime_ratio_threshold = float(datetime_ratio_threshold)
        self.bool_ratio_threshold = float(bool_ratio_threshold)
        self.rare_freq_threshold = float(rare_freq_threshold)
        self.rare_min_count = int(rare_min_count)
        self.high_cardinality_threshold = int(high_cardinality_threshold)
        self.high_cardinality_ratio = float(high_cardinality_ratio)
        self.clip_quantiles = clip_quantiles
        self.max_text_length = int(max_text_length)

        self.training_columns_: List[str] = []
        self.base_to_canonical_: Dict[str, str] = {}
        self.column_types_: Dict[str, str] = {}
        self.drop_columns_: Set[str] = set()
        self.rare_categories_: Dict[str, Set[str]] = {}
        self.seen_categories_: Dict[str, Set[str]] = {}
        self.fuzzy_category_map_: Dict[str, Dict[str, str]] = {}
        self.clip_bounds_: Dict[str, Tuple[float, float]] = {}
        self.log_transform_columns_: Set[str] = set()
        self.missing_indicator_cols_: Set[str] = set()
        self.output_columns_: List[str] = []
        self.audit_report_: Dict[str, Any] = {}

    def _sanitize_frame(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        frame = X.copy()
        if frame.columns.empty:
            raise ValueError("Input dataframe has no columns")

        if fit_mode:
            normalized_names: List[str] = []
            seen: Dict[str, int] = {}
            self.base_to_canonical_ = {}
            for col in frame.columns:
                base = _normalize_column_name_raw(col)
                count = seen.get(base, 0)
                canonical = base if count == 0 else f"{base}_{count}"
                seen[base] = count + 1
                normalized_names.append(canonical)
                if base not in self.base_to_canonical_:
                    self.base_to_canonical_[base] = canonical
            frame.columns = normalized_names
            self.training_columns_ = list(frame.columns)
        else:
            seen_names: Dict[str, int] = {}
            mapped_names: List[str] = []
            for col in frame.columns:
                base = _normalize_column_name_raw(col)
                canonical = self.base_to_canonical_.get(base, base)
                count = seen_names.get(canonical, 0)
                final_name = canonical if count == 0 else f"{canonical}_dup{count}"
                seen_names[canonical] = count + 1
                mapped_names.append(final_name)
            frame.columns = mapped_names
            frame = frame.reindex(columns=self.training_columns_, fill_value=np.nan)

        return frame

    def _infer_type(self, col: str, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        cleaned = _normalize_text_series(series, lowercase=False)
        non_null = cleaned.dropna()
        if non_null.empty:
            return "categorical"

        base_count = max(1, len(non_null))
        bool_ratio = float(_coerce_boolean_series(non_null).notna().mean())
        num_ratio = float(_coerce_numeric_series(non_null).notna().mean())
        dt_ratio = float(_coerce_datetime_series(non_null).notna().mean())
        has_date_hint = any(k in col for k in ("date", "time", "timestamp", "dt"))

        if bool_ratio >= self.bool_ratio_threshold:
            return "boolean"
        if (has_date_hint and dt_ratio >= 0.5) or (dt_ratio >= self.datetime_ratio_threshold):
            return "datetime"
        if num_ratio >= self.numeric_ratio_threshold:
            return "numeric"
        return "categorical"

    def _convert_base(self, col: str, series: pd.Series) -> Any:
        kind = self.column_types_.get(col, "categorical")
        if kind == "boolean":
            return _coerce_boolean_series(series)
        if kind == "numeric":
            return _coerce_numeric_series(series)
        if kind == "datetime":
            return _coerce_datetime_series(series)
        cleaned = _normalize_text_series(series, lowercase=True)
        cleaned = cleaned.map(lambda v: v[: self.max_text_length] if isinstance(v, str) else v)
        return cleaned

    @staticmethod
    def _apply_category_alias(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        compact = re.sub(r"\s+", "", text)
        if text in _COMMON_CATEGORY_ALIASES:
            return _COMMON_CATEGORY_ALIASES[text]
        if compact in _COMMON_CATEGORY_ALIASES:
            return _COMMON_CATEGORY_ALIASES[compact]
        return text

    @staticmethod
    def _build_fuzzy_map(values: pd.Series) -> Dict[str, str]:
        unique_values = [str(v) for v in values.dropna().astype(str).unique().tolist()]
        if len(unique_values) <= 1 or len(unique_values) > 500:
            return {v: v for v in unique_values}

        canonical: List[str] = []
        mapping: Dict[str, str] = {}

        for candidate in sorted(unique_values, key=lambda x: (len(x), x)):
            merged = False
            candidate_compact = candidate.replace(" ", "")
            for anchor in canonical[:150]:
                anchor_compact = anchor.replace(" ", "")
                if candidate_compact == anchor_compact:
                    mapping[candidate] = anchor
                    merged = True
                    break

                if min(len(candidate), len(anchor)) < 4:
                    continue

                score = SequenceMatcher(None, candidate, anchor).ratio()
                if score >= 0.93:
                    mapping[candidate] = anchor
                    merged = True
                    break

            if not merged:
                canonical.append(candidate)
                mapping[candidate] = candidate

        return mapping

    @staticmethod
    def _expand_datetime(col: str, dt: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{col}__year": dt.dt.year.astype("float64"),
                f"{col}__month": dt.dt.month.astype("float64"),
                f"{col}__day": dt.dt.day.astype("float64"),
                f"{col}__weekday": dt.dt.weekday.astype("float64"),
                f"{col}__hour": dt.dt.hour.astype("float64"),
                f"{col}__minute": dt.dt.minute.astype("float64"),
            },
            index=dt.index,
        )

    def _build_fit_metadata(self, frame: pd.DataFrame) -> None:
        row_count = max(1, len(frame))
        drop_cols: Set[str] = set()
        profile: Dict[str, Any] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "column_types": dict(self.column_types_),
            "high_cardinality_columns": [],
            "zero_variance_columns": [],
            "rare_category_columns": {},
            "missing_indicators": [],
        }

        for col in self.training_columns_:
            base = self._convert_base(col, frame[col])
            kind = self.column_types_.get(col, "categorical")

            if kind == "categorical":
                non_null = base.dropna().astype(str)
                nunique = int(non_null.nunique())
                if nunique <= 1:
                    drop_cols.add(col)
                    profile["zero_variance_columns"].append(col)
                    continue
                if nunique > self.high_cardinality_threshold and (nunique / row_count) > self.high_cardinality_ratio:
                    drop_cols.add(col)
                    profile["high_cardinality_columns"].append(col)
                    continue

                if len(non_null) > 0:
                    aliased = non_null.map(self._apply_category_alias)
                    fuzzy_map = self._build_fuzzy_map(aliased)
                    canonicalized = aliased.map(lambda v: fuzzy_map.get(str(v), str(v)) if pd.notna(v) else np.nan)

                    counts = canonicalized.value_counts()
                    min_count = max(self.rare_min_count, int(np.ceil(self.rare_freq_threshold * len(non_null))))
                    rare = set(counts[counts < min_count].index.tolist())
                    self.rare_categories_[col] = rare
                    self.seen_categories_[col] = set(counts.index.tolist())
                    self.fuzzy_category_map_[col] = fuzzy_map
                    if rare:
                        profile["rare_category_columns"][col] = len(rare)
                else:
                    self.rare_categories_[col] = set()
                    self.seen_categories_[col] = set()
                    self.fuzzy_category_map_[col] = {}

            elif kind in {"numeric", "boolean"}:
                num = pd.to_numeric(base, errors="coerce").replace([np.inf, -np.inf], np.nan)
                finite = num.dropna()
                if int(finite.nunique()) <= 1:
                    drop_cols.add(col)
                    profile["zero_variance_columns"].append(col)
                    continue

                if kind == "numeric" and len(finite) >= 10:
                    q_low = float(finite.quantile(self.clip_quantiles[0]))
                    q_high = float(finite.quantile(self.clip_quantiles[1]))
                    if np.isfinite(q_low) and np.isfinite(q_high) and q_low < q_high:
                        self.clip_bounds_[col] = (q_low, q_high)

                    skew = float(finite.skew()) if len(finite) >= 20 else 0.0
                    if abs(skew) >= 2.0 and float(finite.min()) >= 0.0:
                        self.log_transform_columns_.add(col)

            elif kind == "datetime":
                dt = base
                if int(dt.dropna().nunique()) <= 1:
                    drop_cols.add(col)
                    profile["zero_variance_columns"].append(col)

            missing_ratio = float(pd.isna(base).mean())
            if missing_ratio > 0.0 and col not in drop_cols:
                self.missing_indicator_cols_.add(col)

        self.drop_columns_ = drop_cols
        profile["missing_indicators"] = sorted(self.missing_indicator_cols_)
        profile["dropped_columns"] = sorted(self.drop_columns_)
        self.audit_report_ = profile

    def _transform_internal(self, frame: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        for col in self.training_columns_:
            if col in self.drop_columns_:
                continue

            base = self._convert_base(col, frame[col])
            kind = self.column_types_.get(col, "categorical")

            if kind == "datetime":
                dt = _coerce_datetime_series(base)
                dt_features = self._expand_datetime(col, dt)
                out = pd.concat([out, dt_features], axis=1)
                if col in self.missing_indicator_cols_:
                    out[f"{col}__missing"] = dt.isna().astype(int)
                continue

            if kind == "categorical":
                series = _normalize_text_series(base, lowercase=True)
                series = series.map(self._apply_category_alias)
                rare_set = self.rare_categories_.get(col, set())
                seen = self.seen_categories_.get(col, set())
                fuzzy_map = self.fuzzy_category_map_.get(col, {})

                def _map_cat(v: Any) -> Any:
                    if pd.isna(v):
                        return np.nan
                    sv = str(v)
                    sv = fuzzy_map.get(sv, sv)
                    if sv in rare_set:
                        return "__other__"
                    if not fit_mode and seen and sv not in seen:
                        return "__other__"
                    return sv

                series = series.map(_map_cat)
                out[col] = series
                if col in self.missing_indicator_cols_:
                    out[f"{col}__missing"] = series.isna().astype(int)
                continue

            numeric = pd.to_numeric(base, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if kind == "numeric":
                if col in self.clip_bounds_:
                    q_low, q_high = self.clip_bounds_[col]
                    numeric = numeric.clip(lower=q_low, upper=q_high)
                if col in self.log_transform_columns_:
                    numeric = np.log1p(np.clip(numeric, a_min=0.0, a_max=None))

            out[col] = numeric
            if col in self.missing_indicator_cols_:
                out[f"{col}__missing"] = numeric.isna().astype(int)

        out = out.replace([np.inf, -np.inf], np.nan)

        if fit_mode:
            self.output_columns_ = list(out.columns)
        else:
            for col in self.output_columns_:
                if col not in out.columns:
                    out[col] = np.nan
            out = out[self.output_columns_]

        if out.shape[1] == 0:
            raise ValueError("No usable features remain after robust cleaning")

        return out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        frame = self._sanitize_frame(X, fit_mode=True)

        self.column_types_ = {
            col: self._infer_type(col, frame[col])
            for col in self.training_columns_
        }
        self.rare_categories_ = {}
        self.seen_categories_ = {}
        self.fuzzy_category_map_ = {}
        self.clip_bounds_ = {}
        self.log_transform_columns_ = set()
        self.missing_indicator_cols_ = set()

        self._build_fit_metadata(frame)
        _ = self._transform_internal(frame, fit_mode=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.training_columns_:
            raise ValueError("Cleaner is not fitted")
        frame = self._sanitize_frame(X, fit_mode=False)
        out = self._transform_internal(frame, fit_mode=False)
        return out

    def get_audit_report(self) -> Dict[str, Any]:
        report = dict(self.audit_report_)
        report["output_columns"] = list(self.output_columns_)
        report["log_transformed_columns"] = sorted(self.log_transform_columns_)
        report["clip_bounds"] = {k: [float(v[0]), float(v[1])] for k, v in self.clip_bounds_.items()}
        return report

    def run_sanity_checks(self, frame: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Sanity checks expect a pandas DataFrame")

        inf_count = int(np.isinf(frame.select_dtypes(include=[np.number]).to_numpy()).sum()) if frame.shape[1] else 0
        nan_count = int(frame.isna().sum().sum())
        all_nan_cols = [c for c in frame.columns if bool(frame[c].isna().all())]
        return {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "all_nan_columns": all_nan_cols,
        }
