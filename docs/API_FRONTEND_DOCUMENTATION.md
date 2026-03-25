# AutoML API Documentation For Frontend

Base URL (local):

- http://127.0.0.1:8000

Content type:

- JSON for request/response bodies except file upload (multipart/form-data)

API endpoint index:

- GET /
- POST /upload
- POST /eda
- POST /ml-models
- POST /agent/run
- POST /chat
- POST /dashboard/charts
- GET /session/{session_id}/history

---

## 1) Health Check

Endpoint:

- Method: GET
- Path: /

Success response (200):

```json
{
	"message": "Welcome to the AutoML API",
	"status": "ok"
}
```

---

## 2) Upload Dataset

Endpoint:

- Method: POST
- Path: /upload
- Body type: multipart/form-data
- Field: file

Rules:

- Allowed extensions: .csv, .xlsx, .xls
- Max file size: 50 MB
- Minimum dataset size: 10 rows, 2 columns

Success response (200):

```json
{
	"filename": "weatherAUS.csv",
	"shape": {
		"rows": 145460,
		"columns": 24
	},
	"preview": [
		{
			"Date": "2008-12-01",
			"Location": "Albury",
			"MinTemp": 13.4
		}
	],
	"session_id": "session_id_20260324_101530_ab12cd34"
}
```

Common errors:

- 400: unsupported file extension
- 413: file too large
- 422: dataset too small
- 500: upload processing error

---

## 3) Generate EDA Report

Endpoint:

- Method: POST
- Path: /eda

Request body:

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34"
}
```

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"eda_html_path": "http://127.0.0.1:8000/data/session_id_20260324_101530_ab12cd34/index.html"
}
```

---

## 4) Train ML Models

Endpoint:

- Method: POST
- Path: /ml-models

Request body:

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"problem_statement": "Predict if it will rain tomorrow"
}
```

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"problem_type": "classification",
	"target_variable": "RainTomorrow",
	"results": [
		{
			"Model": "RandomForestClassifier",
			"Accuracy": 0.86,
			"Precision": 0.84,
			"Recall": 0.82,
			"F1": 0.83,
			"ROC_AUC": 0.90
		}
	],
	"model_paths": {
		"RandomForestClassifier": "data/datasetAnalysis/session_id_20260324_101530_ab12cd34/RandomForestClassifier.joblib"
	}
}
```

Notes:

- For classification, expect metrics such as Accuracy, Precision, Recall, F1, ROC_AUC.
- For regression, expect metrics such as R2, MAE, RMSE.

---

## 5) Run Full Agent Pipeline

Endpoint:

- Method: POST
- Path: /agent/run

Request body:

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"problem_statement": "Predict if it will rain tomorrow"
}
```

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"status": "ok",
	"problem_type": "classification",
	"target_variable": "RainTomorrow",
	"best_model": "RandomForestClassifier",
	"best_score": 0.86,
	"metric": "Accuracy",
	"all_results": [
		{
			"Model": "RandomForestClassifier",
			"Accuracy": 0.86
		}
	],
	"model_paths": {
		"RandomForestClassifier": "data/datasetAnalysis/session_id_20260324_101530_ab12cd34/RandomForestClassifier.joblib"
	},
	"error_message": null
}
```

---

## 6) Dataset Q&A Chat

Endpoint:

- Method: POST
- Path: /chat

Request body:

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"question": "What is the average rainfall by location?"
}
```

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"question": "What is the average rainfall by location?",
	"answer": "Average rainfall by location is ...",
	"code": "df.groupby('Location')['Rainfall'].mean()",
	"error": null
}
```

Behavior notes:

- Chat is strictly session-scoped using the provided session_id.
- If user asks dataset overview questions (for example: "describe dataset", "dataset description", "dataset overview"), API returns a structured interpreted summary instead of a raw describe-table dump.

---

## 7) Dashboard Charts

Endpoint:

- Method: POST
- Path: /dashboard/charts

Request body:

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"chart_types": ["missing_values", "correlation", "distribution", "bar"]
}
```

Request field details:

- session_id: required
- chart_types: optional list
- Allowed values: distribution, correlation, scatter, bar, missing_values, boxplot
- If chart_types is null or omitted, API returns a curated dashboard

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"columns": ["Date", "Location", "MinTemp", "MaxTemp"],
	"row_count": 145460,
	"charts": [
		{
			"type": "missing_values",
			"figure": {
				"data": [],
				"layout": {}
			}
		},
		{
			"type": "correlation",
			"figure": {
				"data": [],
				"layout": {}
			}
		},
		{
			"type": "distribution",
			"column": "Rainfall",
			"figure": {
				"data": [],
				"layout": {}
			}
		},
		{
			"type": "bar",
			"column": "Location",
			"figure": {
				"data": [],
				"layout": {}
			}
		}
	]
}
```

Dashboard behavior:

- The API returns maximum 4 meaningful charts.
- Charts are sorted by insight priority (missing values, correlation, distribution, bar, scatter, boxplot).
- Frontend can render chart.figure directly using Plotly.js.

---

## 8) Session History By Session ID

Endpoint:

- Method: GET
- Path: /session/{session_id}/history

Example request:

- GET /session/session_id_20260324_101530_ab12cd34/history

Success response (200):

```json
{
	"session_id": "session_id_20260324_101530_ab12cd34",
	"session_path": "data\\datasetAnalysis\\session_id_20260324_101530_ab12cd34",
	"artifacts": {
		"raw_file": true,
		"processed_file": true,
		"eda_report": true,
		"preprocessing": true,
		"trained_models": true
	},
	"dataset_summary": {
		"raw_shape": {
			"rows": 145460,
			"columns": 24
		},
		"processed_shape": {
			"rows": 123456,
			"columns": 36
		}
	},
	"eda": {
		"available": true,
		"url": "http://127.0.0.1:8000/data/session_id_20260324_101530_ab12cd34/index.html"
	},
	"models": {
		"count": 4,
		"names": ["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
		"files": [
			"LinearRegression.joblib",
			"Ridge.joblib",
			"Lasso.joblib",
			"ElasticNet.joblib",
			"preprocessing.joblib"
		]
	},
	"files": [
		{
			"name": "processed_file.csv",
			"size_bytes": 1234567,
			"modified_at": "2026-03-24T11:10:00"
		}
	]
}
```

Common errors:

- 404: session_id not found
- 500: unable to read session artifacts

---

## Standard Error Shape

For API failures, FastAPI returns:

```json
{
	"detail": "Error message"
}
```

Common status codes:

- 400: bad input
- 413: payload too large
- 422: validation error
- 500: internal server error

---

## 9) Pipeline Quality Fix Log And Baseline Checkpoints

This project includes targeted data-pipeline hardening to reduce leakage and improve stability.

Implemented pipeline fixes:

- Train-only feature selection (no full-dataset selection before split).
- Train-only preprocessing fit (imputation, encoding, scaling).
- Replaced feature label encoding with one-hot encoding for categorical features.
- Removed upload-time global row drop based on missing values.
- Added duplicate and target-integrity safeguards in preprocessing.
- Added deterministic baseline runner mode that disables LLM feature selection.

Baseline utility:

- Script: src/evaluation/baseline_runner.py
- Output artifact per session: data/datasetAnalysis/{session_id}/baseline_metrics.json

Deterministic baseline command examples:

```bash
python src/evaluation/baseline_runner.py --session-id session_id_20260323_193846_8802caca --target-col RainTomorrow --problem-type classification --cv 2 --max-rows 1500
python src/evaluation/baseline_runner.py --session-id session_id_20260323_194939_04e09a31 --target-col Price --problem-type regression --cv 2 --max-rows 1500
```

Current checkpoint snapshots:

- Session: session_id_20260323_193846_8802caca
	- problem_type: classification
	- target: RainTomorrow
	- best_model: LogisticRegression
	- best_metric: Accuracy = 0.875
	- balanced_accuracy: 0.772

- Session: session_id_20260323_194939_04e09a31
	- problem_type: regression
	- target: Price
	- best_model: ElasticNet
	- best_metric: R2_Score = 0.7132

Checkpoint comparison table:

| Session ID | Problem Type | Target | Best Model | Primary Metric | Score | Secondary Metric |
|---|---|---|---|---|---:|---|
| session_id_20260323_193846_8802caca | classification | RainTomorrow | LogisticRegression | Accuracy | 0.8750 | Balanced_Accuracy=0.7720 |
| session_id_20260323_194939_04e09a31 | regression | Price | ElasticNet | R2_Score | 0.7132 | RMSE=2366.9585 |

Notes:

- Baseline runs are intended as reproducible checkpoints for before/after comparisons.
- Use the same target, session, cv, and max_rows values for fair comparisons.

Issue-closure matrix (strict mapping):

| Category | Status | Implemented Fix | Evidence |
|---|---|---|---|
| Data leakage: pre-split feature selection | Closed | Feature selection moved to train-only path after split; applied to both train/test | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [src/Regression/regression.py](src/Regression/regression.py), [tests/test_pipeline_integrity.py](tests/test_pipeline_integrity.py) |
| Missing values handling | Closed | Removed global upload-time row drop; moved to train-time imputation in preprocessing pipeline | [src/dataCleaning/featureEngineering01.py](src/dataCleaning/featureEngineering01.py), [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [src/Regression/regression.py](src/Regression/regression.py) |
| Duplicate leakage risk | Closed | Added duplicate removal pre-split and train/test fingerprint overlap detection warning | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [src/Regression/regression.py](src/Regression/regression.py), [tests/test_pipeline_integrity.py](tests/test_pipeline_integrity.py) |
| Categorical encoding mismatch | Closed | Replaced feature label encoding with OneHotEncoder(handle_unknown='ignore') via ColumnTransformer | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [src/Regression/regression.py](src/Regression/regression.py), [tests/test_pipeline_integrity.py](tests/test_pipeline_integrity.py) |
| Class imbalance under-reporting | Closed | Added Balanced_Accuracy metric and class_weight tuning options for major classifiers | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [data/datasetAnalysis/session_id_20260323_193846_8802caca/baseline_metrics.json](data/datasetAnalysis/session_id_20260323_193846_8802caca/baseline_metrics.json) |
| Data integrity: target/features alignment | Closed | Added guards for missing target rows, target-in-features leak checks, and empty-feature matrix checks | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [src/Regression/regression.py](src/Regression/regression.py) |
| String-label target encoding bug | Closed | Fixed classifier to encode any non-numeric target dtype (object/category/string) | [src/Classifier/MLClassifier.py](src/Classifier/MLClassifier.py), [tests/test_pipeline_integrity.py](tests/test_pipeline_integrity.py) |
| Baseline reproducibility and LLM rate-limit fragility | Closed | Added deterministic baseline mode (no LLM feature selection), explicit target support, cv/max_rows controls | [src/evaluation/baseline_runner.py](src/evaluation/baseline_runner.py), [data/datasetAnalysis/session_id_20260323_193846_8802caca/baseline_metrics.json](data/datasetAnalysis/session_id_20260323_193846_8802caca/baseline_metrics.json), [data/datasetAnalysis/session_id_20260323_194939_04e09a31/baseline_metrics.json](data/datasetAnalysis/session_id_20260323_194939_04e09a31/baseline_metrics.json) |

Open residual risks:

- Model quality still depends on target selection quality when users do not provide explicit target_col/problem_type.
- Baseline metrics are sampled when max_rows is set; for final offline validation use full data with max_rows=0.

Reproducibility commands:

Quick checkpoint runs (fast, sampled):

```bash
python src/evaluation/baseline_runner.py --session-id session_id_20260323_193846_8802caca --target-col RainTomorrow --problem-type classification --cv 2 --max-rows 1200
python src/evaluation/baseline_runner.py --session-id session_id_20260323_194939_04e09a31 --target-col Price --problem-type regression --cv 2 --max-rows 1500
```

Full-data validation runs (slower, production-style checkpoint):

```bash
python src/evaluation/baseline_runner.py --session-id session_id_20260323_193846_8802caca --target-col RainTomorrow --problem-type classification --cv 3 --max-rows 0
python src/evaluation/baseline_runner.py --session-id session_id_20260323_194939_04e09a31 --target-col Price --problem-type regression --cv 3 --max-rows 0
```

Expected output artifacts:

- data/datasetAnalysis/session_id_20260323_193846_8802caca/baseline_metrics.json
- data/datasetAnalysis/session_id_20260323_194939_04e09a31/baseline_metrics.json

