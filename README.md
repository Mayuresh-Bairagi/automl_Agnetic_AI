# AutoML Agnetic AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

AutoML backend for tabular datasets with session-based artifacts, automated preprocessing, model training, EDA, agent orchestration, dataset Q&A, and dashboard chart generation.

## Project Summary

| Field | Value |
|---|---|
| Project Name | AutoML Agnetic AI |
| Author | Mayuresh Bairagi |
| Stack | FastAPI, scikit-learn, LangChain, LangGraph |
| Package | auto_ml_model |

## Current Capabilities

- Upload CSV/Excel datasets with size and schema validation.
- Run automated feature engineering and store session artifacts.
- Generate EDA reports (HTML).
- Train multiple models for classification and regression.
- Run an end-to-end LangGraph AutoML agent pipeline.
- Ask natural-language questions over a session dataset.
- Generate interactive dashboard chart payloads.

## Training Pipeline Safety Features

The training flow includes leakage and quality guards that were added to improve reliability:

- Target validation before model training:
    - target column must exist,
    - classification target must have at least 2 classes,
    - regression target must be sufficiently numeric.
- Train-only feature selection and train-only noisy feature pruning.
- Protected target-like column handling in feature engineering.
- Duplicate overlap detection between train and test with enforced removal.
- Imbalance-aware model selection for classification (balanced accuracy based search/evaluation).

## Session Artifacts

Each uploaded dataset is processed into a session folder under:

data/datasetAnalysis/<session_id>/

Typical artifacts:

- raw_file.csv
- processed_file.csv
- preprocessing.joblib
- model .joblib files (per trained model)
- index.html (EDA report, when generated)
- baseline_metrics.json (when baseline runner is used)

## Quick Start

### 1) Install dependencies

```bash
git clone https://github.com/Mayuresh-Bairagi/automl_Agnetic_AI.git
cd automl_Agnetic_AI
pip install -r requirements.txt
```

### 2) Configure environment

Create a .env file in project root:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
LLM_PROVIDER=groq
```

### 3) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

or:

```bash
python app/main.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| / | GET | Health check |
| /upload | POST | Upload dataset and generate session artifacts |
| /eda | POST | Generate EDA report URL for session |
| /ml-models | POST | Detect target and train ML models |
| /agent/run | POST | Run LangGraph end-to-end agent pipeline |
| /chat | POST | Dataset Q&A using generated pandas code |
| /dashboard/charts | POST | Generate dashboard chart specifications |
| /session/{session_id}/history | GET | Return session artifacts and metadata |

## Baseline Runner (Reproducible Benchmark)

Use the baseline runner to generate deterministic benchmark artifacts in baseline_metrics.json.

### Classification example

```bash
python src/evaluation/baseline_runner.py \
    --session-id <session_id> \
    --target-col <target_column> \
    --problem-type classification \
    --cv 2 \
    --max-rows 3000
```

### Regression example

```bash
python src/evaluation/baseline_runner.py \
    --session-id <session_id> \
    --target-col <target_column> \
    --problem-type regression \
    --cv 2 \
    --max-rows 3000
```

## Tests

Run integrity tests:

```bash
pytest tests/test_pipeline_integrity.py -q -rA
```

## Key Dependencies

| Category | Packages |
|---|---|
| API | fastapi, uvicorn, python-multipart |
| Data/ML | pandas, numpy, scikit-learn, xgboost, lightgbm, catboost |
| Agent/LLM | langchain, langchain_community, langchain_google_genai, langchain_groq, langgraph |
| Visualization | plotly, ydata-profiling, seaborn, matplotlib |
| Utilities | joblib, openpyxl, python-dotenv, structlog |

## Notes

- For classification with rare classes, stratified train/test split can fail if a class has fewer than 2 samples.
- Use explicit target selection for benchmarking to avoid invalid inferred targets.

## License

No LICENSE file is currently included in this repository.
