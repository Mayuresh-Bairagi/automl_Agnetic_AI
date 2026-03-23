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

