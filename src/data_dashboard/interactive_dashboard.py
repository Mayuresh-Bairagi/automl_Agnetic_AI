"""
Interactive Dashboard
=====================
Generates Plotly-powered, Power-BI-style chart specifications for an uploaded
dataset session.

All charts are returned as JSON-serialisable dicts (``plotly.io.to_json``
compatible) so the frontend can render them directly with Plotly.js.

Available charts
----------------
- distribution   : histogram + KDE for each numeric column
- correlation    : Pearson correlation heatmap
- scatter        : scatter plot matrix (up to 5 numeric columns)
- bar            : top-N value counts for each categorical column
- boxplot        : box-and-whisker for numeric columns grouped by a category
- missing_values : horizontal bar chart showing % missing per column
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from expection.customExpection import AutoML_Exception
from logger.customlogger import CustomLogger

log = CustomLogger().get_logger(__name__)

_MAX_CATEGORIES = 15   # cap categorical bar charts
_MAX_SCATTER_COLS = 5  # cap scatter matrix columns
_MAX_BOXPLOT_COLS = 8  # cap boxplot columns to keep response size manageable


class InteractiveDashboard:
    """
    Build an interactive dashboard for a processed session dataset.

    Parameters
    ----------
    session_id : str
        Session identifier created by the /upload endpoint.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._df: Optional[pd.DataFrame] = None
        log.info("InteractiveDashboard initialised", session_id=session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        csv_path = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", self.session_id, "processed_file.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"No processed dataset found for session: {self.session_id}"
            )
        self._df = pd.read_csv(csv_path)
        log.info(
            "Dataset loaded for dashboard",
            session_id=self.session_id,
            shape=str(self._df.shape),
        )
        return self._df

    @staticmethod
    def _fig_to_dict(fig: go.Figure) -> Dict[str, Any]:
        """Convert a Plotly Figure to a JSON-serialisable dict."""
        return json.loads(fig.to_json())

    # ------------------------------------------------------------------
    # Individual chart builders
    # ------------------------------------------------------------------

    def _chart_distributions(self, df: pd.DataFrame) -> List[Dict]:
        """Histogram (+ optional KDE) for each numeric column."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        charts = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            fig = px.histogram(
                df,
                x=col,
                nbins=40,
                marginal="box",
                title=f"Distribution — {col}",
                template="plotly_white",
            )
            charts.append({"type": "distribution", "column": col, "figure": self._fig_to_dict(fig)})
        return charts

    def _chart_correlation(self, df: pd.DataFrame) -> Optional[Dict]:
        """Pearson correlation heatmap for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        corr = numeric_df.corr(numeric_only=True).round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix",
            template="plotly_white",
            aspect="auto",
        )
        return {"type": "correlation", "figure": self._fig_to_dict(fig)}

    def _chart_scatter_matrix(self, df: pd.DataFrame) -> Optional[Dict]:
        """Scatter-plot matrix for up to _MAX_SCATTER_COLS numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = numeric_cols[:_MAX_SCATTER_COLS]
        if len(cols) < 2:
            return None
        fig = px.scatter_matrix(
            df,
            dimensions=cols,
            title="Scatter-Plot Matrix",
            template="plotly_white",
        )
        fig.update_traces(diagonal_visible=False)
        return {"type": "scatter_matrix", "columns": cols, "figure": self._fig_to_dict(fig)}

    def _chart_bar_categorical(self, df: pd.DataFrame) -> List[Dict]:
        """Top-N value counts for each low-cardinality categorical column."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        charts = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > _MAX_CATEGORIES or n_unique < 2:
                continue
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(
                counts,
                x=col,
                y="count",
                title=f"Value Counts — {col}",
                template="plotly_white",
                color="count",
                color_continuous_scale="Blues",
            )
            charts.append({"type": "bar", "column": col, "figure": self._fig_to_dict(fig)})
        return charts

    def _chart_missing_values(self, df: pd.DataFrame) -> Optional[Dict]:
        """Horizontal bar chart showing % missing per column (only missing cols)."""
        missing_pct = (df.isnull().mean() * 100).round(2)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        if missing_pct.empty:
            return None
        fig = px.bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation="h",
            title="Missing Values (%)",
            labels={"x": "Missing (%)", "y": "Column"},
            template="plotly_white",
            color=missing_pct.values,
            color_continuous_scale="Reds",
        )
        return {"type": "missing_values", "figure": self._fig_to_dict(fig)}

    def _chart_boxplot(self, df: pd.DataFrame) -> List[Dict]:
        """Box plots for numeric columns (one figure per column)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        charts = []
        for col in numeric_cols[:_MAX_BOXPLOT_COLS]:
            fig = px.box(
                df,
                y=col,
                title=f"Box Plot — {col}",
                template="plotly_white",
                points="outliers",
            )
            charts.append({"type": "boxplot", "column": col, "figure": self._fig_to_dict(fig)})
        return charts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_charts(self, chart_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build and return all requested chart types.

        Parameters
        ----------
        chart_types : list[str] | None
            Subset of ``["distribution", "correlation", "scatter", "bar",
            "missing_values", "boxplot"]``.  Pass *None* to get all charts.

        Returns
        -------
        dict
          ``session_id``, ``columns`` (list), ``charts`` (list of chart dicts).
        """
        try:
            df = self._load_df()
            all_types = {"distribution", "correlation", "scatter", "bar", "missing_values", "boxplot"}
            requested = set(chart_types) if chart_types else all_types

            charts: List[Dict] = []

            if "distribution" in requested:
                charts.extend(self._chart_distributions(df))
            if "correlation" in requested:
                chart = self._chart_correlation(df)
                if chart:
                    charts.append(chart)
            if "scatter" in requested:
                chart = self._chart_scatter_matrix(df)
                if chart:
                    charts.append(chart)
            if "bar" in requested:
                charts.extend(self._chart_bar_categorical(df))
            if "missing_values" in requested:
                chart = self._chart_missing_values(df)
                if chart:
                    charts.append(chart)
            if "boxplot" in requested:
                charts.extend(self._chart_boxplot(df))

            log.info(
                "Dashboard charts generated",
                session_id=self.session_id,
                chart_count=len(charts),
            )

            return {
                "session_id": self.session_id,
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "charts": charts,
            }

        except Exception as exc:
            log.error("Dashboard generation failed", error=str(exc))
            raise AutoML_Exception("Dashboard generation failed", exc)
