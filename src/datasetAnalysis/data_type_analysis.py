import pandas as pd
import time
from expection.customExpection import AutoML_Exception
from model.models import *
from utils.model_loader import ModelLoader
from logger.customlogger import CustomLogger
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from Propmt.propmt_lib import PROMPT_REGISTRY
from typing import List, Dict


class DataTypeAnalyzer:
    def __init__(self, dataset):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            self.parser = JsonOutputParser(pydantic_object=ColumnRecommendation)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.propmt = PROMPT_REGISTRY['change_data_type']
            self.df = dataset

            self.log.info("Data type analysis initialized successfully")

        except Exception as e:
            self.log.error(f"Error initializing data type analysis ",error = e)
            raise AutoML_Exception(f"Error initializing data type analysis: {e}")

    @staticmethod
    def _coerce_numeric_like(series: pd.Series) -> pd.Series:
        """Convert numeric-like text values into numeric values.

        Handles comma-separated numbers, currency markers, unit text, and
        surrounding whitespace while preserving non-convertible rows as NaN.
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        normalized = series.astype("string").str.strip()
        normalized = normalized.str.replace(",", "", regex=False)
        normalized = normalized.str.replace(r"[^0-9.\-]+", "", regex=True)
        normalized = normalized.replace({"": pd.NA, "-": pd.NA, ".": pd.NA, "-.": pd.NA})
        return pd.to_numeric(normalized, errors="coerce")

    def get_column_info(self, df, sample_size=5):
        info = {}
        for col in df.columns:
            info[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().astype(str).sample(min(sample_size, len(df))).tolist()
            }
        return info

    def analyze_data_type(self):
        try:
            column_meta_data = self.get_column_info(self.df)
            chain = self.propmt | self.llm | self.fixing_parser

            self.log.info("Meta-data analysis chain initialized")

            response = self._invoke_with_retry(
                chain,
                {
                    'return_instructions': self.parser.get_format_instructions(),
                    'Column_metadata': column_meta_data
                },
            )

            self.log.info("Metadata extraction successful", keys=response)
            return response

        except Exception as e:
            message = str(e).lower()
            is_rate_limited = (
                "429" in message
                or "resource_exhausted" in message
                or "rate" in message and "limit" in message
                or "quota" in message
            )
            if is_rate_limited:
                self.log.warning("LLM rate-limited; using deterministic dtype fallback", error=str(e))
                return self._fallback_recommendations(self.df)

            self.log.error(f"Error during data type analysis: {e}")
            raise AutoML_Exception(f"Error during data type analysis: {e}")

    def _fallback_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        recommendations: List[Dict] = []
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            suggested_dtype = "object"
            reason = "Defaulted to object"

            if pd.api.types.is_bool_dtype(series):
                suggested_dtype = "boolean"
                reason = "Detected boolean dtype"
            else:
                numeric = self._coerce_numeric_like(non_null)
                numeric_ratio = float(numeric.notna().mean()) if len(non_null) else 0.0
                if numeric_ratio >= 0.9 and len(non_null) > 0:
                    is_integer_like = (numeric.dropna() % 1 == 0).all()
                    suggested_dtype = "integer" if is_integer_like else "float"
                    reason = "Most values parse as numeric"
                else:
                    dt = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
                    dt_ratio = float(dt.notna().mean()) if len(non_null) else 0.0
                    if dt_ratio >= 0.9 and len(non_null) > 0:
                        suggested_dtype = "date"
                        reason = "Most values parse as datetime"

            recommendations.append(
                {
                    "column_name": str(col),
                    "suggested_dtype": suggested_dtype,
                    "reason": reason,
                }
            )

        return recommendations

    def _invoke_with_retry(self, chain, payload, max_attempts: int = 5):
        for attempt in range(1, max_attempts + 1):
            try:
                return chain.invoke(payload)
            except Exception as e:
                message = str(e).lower()
                is_rate_limited = (
                    "429" in message
                    or "resource_exhausted" in message
                    or "rate" in message and "limit" in message
                    or "quota" in message
                )
                if (not is_rate_limited) or attempt == max_attempts:
                    raise

                wait_seconds = min(30, 2 ** attempt)
                self.log.warning(
                    "Rate limit from LLM; retrying",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    wait_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)

    def generate_conversion_code(self, recommendations: List[Dict]) -> str:
        code_lines = ["import pandas as pd"]
        for col in recommendations:
            name = col['column_name']
            suggested = col['suggested_dtype']

            if suggested == "date":
                code_lines.append(
                    f'df["{name}"] = pd.to_datetime(df["{name}"], errors="coerce", dayfirst=True)'
                )
            elif suggested == "time":
                code_lines.append(
                    f'df["{name}"] = pd.to_datetime(df["{name}"], errors="coerce").dt.time'
                )
            elif suggested in ["integer", "int"]:
                code_lines.append(f'df["{name}"] = df["{name}"].astype(int)')
            elif suggested == "float":
                code_lines.append(f'df["{name}"] = df["{name}"].astype(float)')
            else:
                code_lines.append(f'# {name} remains as {suggested}')

        return "\n".join(code_lines)

    def apply_conversions(self, df: pd.DataFrame, recommendations: List[Dict]) -> pd.DataFrame:
        for col in recommendations:
            name = col['column_name']
            suggested = col['suggested_dtype']

            try:
                if name not in df.columns:
                    continue

                if suggested == "date":
                    df[name] = pd.to_datetime(df[name], errors="coerce", dayfirst=True)
                elif suggested == "time":
                    df[name] = pd.to_datetime(df[name], errors="coerce").dt.time
                elif suggested in ["integer", "int"]:
                    numeric = self._coerce_numeric_like(df[name])
                    # Apply conversion only when majority of rows are numeric-like.
                    ratio = float(numeric.notna().mean()) if len(df[name]) else 0.0
                    if ratio >= 0.7:
                        df[name] = numeric.round().astype("Int64")
                    else:
                        self.log.warning(
                            "Skipping integer conversion due to low numeric parse ratio",
                            column=name,
                            parse_ratio=round(ratio, 4),
                        )
                elif suggested == "float":
                    numeric = self._coerce_numeric_like(df[name])
                    ratio = float(numeric.notna().mean()) if len(df[name]) else 0.0
                    if ratio >= 0.7:
                        df[name] = numeric.astype("Float64")
                    else:
                        self.log.warning(
                            "Skipping float conversion due to low numeric parse ratio",
                            column=name,
                            parse_ratio=round(ratio, 4),
                        )
                else:
                    pass
            except Exception as e:
                self.log.warning(f"Could not convert column {name} to {suggested}: {e}")

        return df




if __name__ == "__main__":
    analyzer = DataTypeAnalyzer(r"D:\College\Project\automl\data\Data_Train.csv")

    recommendations = analyzer.analyze_data_type()
    if isinstance(recommendations, dict) and "columns" in recommendations:
        recommendations = recommendations["columns"]  

    code_snippet = analyzer.generate_conversion_code(recommendations)
    print("Generated Conversion Code ")
    print(code_snippet)

    # Apply conversions directly
    converted_df = analyzer.apply_conversions(analyzer.df, recommendations)
    print("Converted DataFrame (head)")
    print(converted_df.info())
    #print(converted_df.head())

