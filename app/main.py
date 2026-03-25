from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from io import BytesIO
from datetime import datetime
import pandas as pd
import uvicorn

from src.dataCleaning.featureEngineering01 import FeatureEngineer1
from src.data_dashboard.eda import EDA
from src.data_dashboard.interactive_dashboard import InteractiveDashboard
from src.Regression.regression import AutoMLRegressor
from model.models import (
    requestEDA,
    request_ml_models,
    AgentRunRequest,
    AgentRunResponse,
    QARequest,
    QAResponse,
    DashboardRequest,
    DashboardResponse,
)
from src.problem_statement.target_variable import TargetVariable
from src.Classifier.MLClassifier import AutoMLClassifier
from src.agent.automl_agent import AutoMLAgent
from src.data_qa.dataset_qa import DatasetQA

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB
_MIN_ROWS = 10
_MIN_COLS = 2
_ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="AutoML API",
    description=(
        "Enterprise-grade AutoML backend: upload a dataset, run EDA, "
        "train multiple ML models with automated hyperparameter tuning, "
        "and interact with your data through a natural-language Q&A interface."
    ),
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

folder_path = Path("data/datasetAnalysis")
folder_path.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=folder_path), name="data")



@app.get("/")
async def root():
    """Health-check endpoint."""
    return {"message": "Welcome to the AutoML API", "status": "ok"}


@app.post("/upload", summary="Upload a CSV or Excel dataset and trigger feature engineering")
async def upload_file(file: UploadFile = File(...)):
    """Accept a CSV or Excel file, run automated feature engineering, and return
    a ``session_id`` that must be passed to all subsequent endpoints.

    Limits
    ------
    * Maximum file size: 50 MB
    * Minimum dataset: 10 rows × 2 columns
    * Supported formats: ``.csv``, ``.xlsx``, ``.xls``
    """
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix}'. "
                    f"Accepted formats: {', '.join(_ALLOWED_EXTENSIONS)}"
                ),
            )

        contents = await file.read()

        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {_MAX_UPLOAD_BYTES // (1024*1024)} MB.",
            )

        if suffix == ".csv":
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))

        if df.shape[0] < _MIN_ROWS:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset has only {df.shape[0]} rows; at least {_MIN_ROWS} are required.",
            )
        if df.shape[1] < _MIN_COLS:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset has only {df.shape[1]} columns; at least {_MIN_COLS} are required.",
            )

        fe = FeatureEngineer1(df)
        df, session_id = fe.generate_features()

        preview = df.head(10).to_dict(orient="records")

        return {
            "filename": file.filename,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "preview": preview,
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/eda", summary="Generate an interactive HTML EDA report for a session")
async def eda(request: requestEDA):
    """Generate a `ydata-profiling` HTML report for the processed dataset
    associated with *session_id* and return its URL.
    """
    try:
        session_id = request.session_id
        eda_obj = EDA(session_id=session_id)
        html_path = eda_obj.generate_report()

        html_url = f"http://127.0.0.1:8000/data/{session_id}/index.html"

        return {"session_id": session_id, "eda_html_path": html_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")


@app.post("/ml-models", summary="Train ML models for a session")
async def ml_model(request: request_ml_models):
    """Detect the target variable, select features, train a suite of models with
    GridSearchCV, and return per-model evaluation metrics.

    The response includes additional metrics beyond accuracy/R² (precision,
    recall, ROC-AUC for classifiers; MAE, RMSE for regressors) as well as
    cross-validation scores for a more robust performance estimate.
    """
    try:
        session_id = request.session_id
        problem_statement = request.problem_statement
        target_var_handler = TargetVariable(session_id=session_id)
        result, df = target_var_handler.get_target_variable(problem_statement)

        problem_statement_type = result["problem_type"]
        target_col = result.get("target_variable")

        if target_col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Target column '{target_col}' is not present in processed dataset.",
            )

        if problem_statement_type.lower() == "classification":
            class_count = int(df[target_col].nunique(dropna=True))
            if class_count < 2:
                raise HTTPException(
                    status_code=422,
                    detail=f"Classification target '{target_col}' has fewer than 2 classes.",
                )

        if problem_statement_type.lower() == "regression":
            numeric_target = pd.to_numeric(df[target_col], errors="coerce")
            valid_ratio = float(numeric_target.notna().mean()) if len(numeric_target) else 0.0
            if valid_ratio < 0.8:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Regression target '{target_col}' is not sufficiently numeric "
                        f"(valid ratio={valid_ratio:.2f})."
                    ),
                )

        if problem_statement_type.lower() == "regression":
            automl_regressor = AutoMLRegressor(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df,
            )
            results_df, trained_models, model_paths = automl_regressor.train_models(
                skip_heavy=True
            )

        elif problem_statement_type.lower() == "classification":
            automl_classifier = AutoMLClassifier(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df,
            )
            results_df, trained_models, model_paths = automl_classifier.train_models(
                skip_heavy=True
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported problem type: {problem_statement_type}",
            )

        results_dict = results_df.to_dict(orient="records")

        return {
            "session_id": session_id,
            "problem_type": problem_statement_type,
            "target_variable": result.get("target_variable"),
            "results": results_dict,
            "model_paths": model_paths,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


# ---------------------------------------------------------------------------
# Agentic pipeline endpoint
# ---------------------------------------------------------------------------

@app.post("/agent/run", response_model=AgentRunResponse, summary="Run the full AutoML pipeline via LangGraph agent")
async def agent_run(request: AgentRunRequest):
    """
    Orchestrate the complete AutoML pipeline (data loading → target detection →
    model training → report) using a LangGraph stateful agent.

    The agent progresses through nodes autonomously, with conditional routing
    that halts gracefully on errors and returns a structured report.
    """
    try:
        agent = AutoMLAgent()
        final_state = agent.run(
            session_id=request.session_id,
            problem_statement=request.problem_statement,
        )

        if final_state.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=final_state.get("error_message", "Agent encountered an error"),
            )

        report = final_state.get("report") or {}
        return AgentRunResponse(
            session_id=request.session_id,
            status="ok",
            problem_type=report.get("problem_type"),
            target_variable=report.get("target_variable"),
            best_model=report.get("best_model"),
            best_score=report.get("best_score"),
            metric=report.get("metric"),
            all_results=report.get("all_results"),
            model_paths=report.get("model_paths"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {str(e)}")


# ---------------------------------------------------------------------------
# Dataset Q&A endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=QAResponse, summary="Ask a natural language question about your dataset")
async def chat(request: QARequest):
    """
    Answer a natural language question over the uploaded dataset for the given
    session.  The LLM generates a pandas snippet which is executed in a safe
    sandbox; the computed result is returned alongside the generated code so
    users can inspect and trust the answer.
    """
    try:
        qa = DatasetQA(session_id=request.session_id)
        result = qa.answer(request.question)

        return QAResponse(
            session_id=request.session_id,
            question=result["question"],
            answer=result["answer"],
            code=result["code"],
            error=result["error"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


# ---------------------------------------------------------------------------
# Interactive dashboard endpoint
# ---------------------------------------------------------------------------

@app.post("/dashboard/charts", response_model=DashboardResponse, summary="Generate interactive Power-BI style charts")
async def dashboard_charts(request: DashboardRequest):
    """
    Generate a compact set of interactive Plotly chart specifications for the
    uploaded dataset. The frontend can render these directly with Plotly.js.

    Supported chart types: ``distribution``, ``correlation``, ``scatter``,
    ``bar``, ``missing_values``, ``boxplot``.  Pass ``chart_types=null`` to
    receive a curated dashboard (maximum 4 meaningful charts).
    """
    try:
        dashboard = InteractiveDashboard(session_id=request.session_id)
        result = dashboard.get_charts(chart_types=request.chart_types)

        return DashboardResponse(
            session_id=result["session_id"],
            columns=result["columns"],
            row_count=result["row_count"],
            charts=result["charts"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Session history endpoint
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}/history", summary="Get full history and artifacts for a session")
async def session_history(session_id: str):
    """
    Return a consolidated view of all known artifacts generated for a session.

    Includes data file presence, dataset shapes, EDA report URL, trained model
    files, and file-level metadata so frontend can render a session timeline.
    """
    try:
        session_path = folder_path / session_id
        if not session_path.exists() or not session_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        raw_file = session_path / "raw_file.csv"
        processed_file = session_path / "processed_file.csv"
        eda_file = session_path / "index.html"

        raw_shape = None
        processed_shape = None

        if raw_file.exists():
            try:
                raw_df = pd.read_csv(raw_file)
                raw_shape = {"rows": int(raw_df.shape[0]), "columns": int(raw_df.shape[1])}
            except Exception:
                raw_shape = None

        if processed_file.exists():
            try:
                processed_df = pd.read_csv(processed_file)
                processed_shape = {
                    "rows": int(processed_df.shape[0]),
                    "columns": int(processed_df.shape[1]),
                }
            except Exception:
                processed_shape = None

        model_files = sorted([p.name for p in session_path.glob("*.joblib")])
        trained_models = [
            p.replace(".joblib", "")
            for p in model_files
            if p.lower() != "preprocessing.joblib"
        ]

        files = []
        for f in sorted(session_path.iterdir(), key=lambda x: x.name.lower()):
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size_bytes": int(f.stat().st_size),
                        "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                )

        return {
            "session_id": session_id,
            "session_path": str(session_path),
            "artifacts": {
                "raw_file": raw_file.exists(),
                "processed_file": processed_file.exists(),
                "eda_report": eda_file.exists(),
                "preprocessing": "preprocessing.joblib" in model_files,
                "trained_models": bool(trained_models),
            },
            "dataset_summary": {
                "raw_shape": raw_shape,
                "processed_shape": processed_shape,
            },
            "eda": {
                "available": eda_file.exists(),
                "url": f"http://127.0.0.1:8000/data/{session_id}/index.html" if eda_file.exists() else None,
            },
            "models": {
                "count": len(trained_models),
                "names": trained_models,
                "files": model_files,
            },
            "files": files,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session history: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
