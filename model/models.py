from pydantic import BaseModel , RootModel
from typing import List , Union, Literal,Optional
from enum import Enum


class ColumnRecommendation(BaseModel):
    column_name: str
    current_dtype: str
    sample_values : list
    suggested_dtype: Literal["object", "integer", "float", "date", "boolean"]
    reason: str

class DataTypeRecommendation(BaseModel):
    columns: List[ColumnRecommendation]

class FeatureEngineering(BaseModel):
    remake: Literal["yes", "no"]
    code: str


class requestEDA(BaseModel):
    session_id: str


class TargetVariableRecommendation(BaseModel):
    target_variable: str
    problem_type: Literal["regression", "classification", "clustering"]
    justification: str

class RankedFeature(BaseModel):
    name: str
    score: float
    reason: Optional[str]

class FeatureSelectionOutput(BaseModel):
    target_col : str
    selected_features: List[str]
    dropped_features: List[str]
    ranked_features: List[RankedFeature]


class request_ml_models(BaseModel):
    session_id : str
    problem_statement : str


# ---------------------------------------------------------------------------
# Agentic pipeline
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    """Request body for the /agent/run endpoint."""
    session_id: str
    problem_statement: str


class AgentRunResponse(BaseModel):
    """Response from the /agent/run endpoint."""
    session_id: str
    status: str
    problem_type: Optional[str] = None
    target_variable: Optional[str] = None
    best_model: Optional[str] = None
    best_score: Optional[Union[float, str]] = None
    metric: Optional[str] = None
    all_results: Optional[List[dict]] = None
    model_paths: Optional[dict] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset Q&A
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    """Request body for the /chat endpoint."""
    session_id: str
    question: str


class QAResponse(BaseModel):
    """Response from the /chat endpoint."""
    session_id: str
    question: str
    answer: Optional[str] = None
    code: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Interactive dashboard
# ---------------------------------------------------------------------------

class DashboardRequest(BaseModel):
    """Request body for the /dashboard/charts endpoint."""
    session_id: str
    chart_types: Optional[List[str]] = None  # None → all charts


class DashboardResponse(BaseModel):
    """Response from the /dashboard/charts endpoint."""
    session_id: str
    columns: List[str]
    row_count: int
    charts: List[dict]


