from __future__ import annotations

from io import BytesIO
import zipfile


def get_usage_notes() -> dict:
    return {
        "text_to_numeric": (
            "Text columns are converted by encoders in preprocessing.joblib "
            "(typically OneHotEncoder)."
        ),
        "numeric_scaling": (
            "Numeric columns may be standardized/scaled by preprocessing before prediction."
        ),
        "classification_decoding": (
            "For classification, target_encoder converts predicted numeric classes "
            "back to original labels."
        ),
        "download_instructions": (
            "Download your selected model, preprocessing.joblib, and its usage-script ZIP; "
            "then run use_model.py locally."
        ),
    }


def build_model_usage_script(model_file_name: str) -> str:
    return f'''from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference using a downloaded AutoML model artifact."
    )
    parser.add_argument("--input", default="input_template.csv", help="Path to local input CSV file")
    parser.add_argument("--model-path", default="{model_file_name}", help="Path to downloaded model .joblib file")
    parser.add_argument("--preprocessing-path", default="preprocessing.joblib", help="Path to downloaded preprocessing .joblib file")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    return parser.parse_args()


def resolve_input_path(cli_value: str) -> str:
    input_path = cli_value.strip()
    if input_path and Path(input_path).exists():
        return input_path
    if not input_path:
        input_path = input("Enter path to your local CSV file: ").strip()
    if not input_path:
        raise ValueError("Input CSV path is required.")
    return input_path


def load_artifacts(model_path: str, preprocessing_path: str):
    model = joblib.load(model_path)
    preprocessing = joblib.load(preprocessing_path)

    cleaner = preprocessing.get("cleaner")
    preprocessor = preprocessing.get("preprocessor")
    target_encoder = preprocessing.get("target_encoder")
    dropped_features = preprocessing.get("dropped_features", [])

    if not isinstance(dropped_features, list):
        dropped_features = []

    if preprocessor is None:
        raise ValueError("Preprocessing artifact is missing 'preprocessor'.")

    return model, cleaner, preprocessor, target_encoder, dropped_features


def describe_preprocessing(cleaner, preprocessor, target_encoder, dropped_features: list[str]) -> None:
    print("Preprocessing summary:")
    print(f"- Dropped features before transform: {{dropped_features}}")
    print(f"- Robust cleaning layer available: {{'yes' if cleaner is not None else 'no'}}")

    transformers = getattr(preprocessor, "transformers_", [])
    for transformer_name, transformer_obj, transformer_cols in transformers:
        if transformer_name == "remainder":
            continue
        readable_type = type(transformer_obj).__name__
        print(f"- Step '{{transformer_name}}': {{readable_type}} on columns {{list(transformer_cols)}}")

    if target_encoder is not None:
        print("- Target decoder available: yes (classification labels will be text values)")
    else:
        print("- Target decoder available: no (regression numeric output)")


def align_input_to_preprocessor(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    expected_cols = list(getattr(preprocessor, "feature_names_in_", []))
    if not expected_cols:
        return df

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[expected_cols]


def prepare_features(input_df: pd.DataFrame, preprocessor, dropped_features: list[str]):
    cleaned = input_df.drop(columns=dropped_features, errors="ignore")
    cleaned = align_input_to_preprocessor(cleaned, preprocessor)
    return preprocessor.transform(cleaned)


def prepare_features_with_cleaner(
    input_df: pd.DataFrame,
    cleaner,
    preprocessor,
    dropped_features: list[str],
):
    cleaned = input_df.drop(columns=dropped_features, errors="ignore")
    if cleaner is not None:
        cleaned = cleaner.transform(cleaned)
    cleaned = align_input_to_preprocessor(cleaned, preprocessor)
    return preprocessor.transform(cleaned)


def predict_values(input_df: pd.DataFrame, model, transformed, target_encoder):
    out_df = input_df.copy()

    if target_encoder is not None:
        pred_encoded = model.predict(transformed)
        out_df["prediction"] = target_encoder.inverse_transform(pred_encoded)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(transformed)
            out_df["confidence"] = np.max(proba, axis=1)
    else:
        out_df["prediction"] = model.predict(transformed)

    return out_df


def save_predictions(out_df: pd.DataFrame, output: str) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()

    print("Step 1/6: Reading input path...")
    input_path = resolve_input_path(args.input)

    print("Step 2/6: Loading local input data...")
    input_df = pd.read_csv(input_path)

    print("Step 3/6: Loading model and preprocessing artifacts...")
    model, cleaner, preprocessor, target_encoder, dropped_features = load_artifacts(
        model_path=args.model_path,
        preprocessing_path=args.preprocessing_path,
    )

    print("Step 4/6: Explaining preprocessing and conversions...")
    describe_preprocessing(cleaner, preprocessor, target_encoder, dropped_features)

    print("Step 5/6: Applying preprocessing and prediction...")
    transformed = prepare_features_with_cleaner(input_df, cleaner, preprocessor, dropped_features)
    out_df = predict_values(input_df, model, transformed, target_encoder)

    print("Step 6/6: Saving predictions...")
    output_path = save_predictions(out_df, args.output)

    print(f"Saved predictions to: {{output_path}}")
    print("Preview:")
    print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
'''


def build_model_usage_readme(model_file_name: str) -> str:
    return f'''# How To Use Downloaded Model

This package helps you run local predictions using the model file `{model_file_name}`.

## Files Included

- `use_model.py`: prediction script
- `requirements.txt`: required Python packages
- `TECHNICAL_NOTES.md`: preprocessing and conversion details
- `{model_file_name}`: trained model artifact (included)
- `preprocessing.joblib`: preprocessing artifact (included)
- `input_template.csv`: optional same-schema starter input (included when available)

## Step-by-Step

1. Extract ZIP and keep files in one folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run with CLI arguments:

```bash
python use_model.py --input my_local_data.csv --model-path {model_file_name} --preprocessing-path preprocessing.joblib --output predictions.csv
```

4. Or run with almost no changes (uses included defaults):

```bash
python use_model.py
```

5. If input path is not provided or not found, script asks interactively:

```bash
python use_model.py
```

## Notes

- Your input CSV must have feature columns expected by the model preprocessing.
- For classification models, output includes label prediction and confidence when available.
- For regression models, output includes numeric prediction only.
- Input data should match training schema features; target column is not required.

## How Text Values Become Numbers

Inside `preprocessing.joblib`, your pipeline stores the conversion logic used during model training.

- Text columns are converted using encoders inside the `preprocessor` (most commonly `OneHotEncoder`).
- Numeric columns are usually scaled/normalized (`StandardScaler` or similar).
- If this is classification, `target_encoder` converts predicted class IDs back to readable labels.

When you run `use_model.py`, it automatically applies the same conversion sequence before prediction.
'''


def build_model_usage_technical_notes() -> str:
    return '''# Technical Notes (Non-Technical Friendly)

This file explains what happens under the hood in simple words.

## 1) Why preprocessing is required

Machine learning models cannot directly use raw text values like city names. The preprocessing object converts your raw data into model-ready numeric arrays.

## 2) Common conversion steps

- Text columns -> numeric vectors (typically OneHotEncoder)
- Numeric columns -> scaled numeric values (typically StandardScaler)
- Selected columns may be removed first (`dropped_features`)

## 3) Classification-specific decoding

For classification models, predictions are often internal numeric IDs. The `target_encoder` converts them back to original text classes like "Yes" / "No".

## 4) Why column names matter

Your input CSV must include the same feature names expected by preprocessing. The script aligns columns and fills missing expected columns with empty values so transform can run.

## 5) Troubleshooting quick map

- Error: missing preprocessor
  Meaning: preprocessing.joblib is incomplete or wrong file.

- Error: feature mismatch
  Meaning: input CSV columns differ too much from model training schema.

- Error: no predictions
  Meaning: model or preprocessing path is incorrect.
'''


def build_usage_zip_bytes(
    model_file_name: str,
    model_bytes: bytes,
    preprocessing_bytes: bytes,
    input_template_csv: str | None = None,
) -> bytes:
    requirements_text = "pandas\nnumpy\njoblib\nscikit-learn\n"
    script_text = build_model_usage_script(model_file_name=model_file_name)
    readme_text = build_model_usage_readme(model_file_name=model_file_name)
    technical_notes_text = build_model_usage_technical_notes()

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("use_model.py", script_text)
        zipf.writestr("requirements.txt", requirements_text)
        zipf.writestr("README.md", readme_text)
        zipf.writestr("TECHNICAL_NOTES.md", technical_notes_text)
        zipf.writestr(model_file_name, model_bytes)
        zipf.writestr("preprocessing.joblib", preprocessing_bytes)
        if input_template_csv:
            zipf.writestr("input_template.csv", input_template_csv)

    return buffer.getvalue()
