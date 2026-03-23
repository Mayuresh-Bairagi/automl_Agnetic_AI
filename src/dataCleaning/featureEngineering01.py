import pandas as pd
import re
import time
from expection.customExpection import AutoML_Exception
from model.models import *
from utils.model_loader import ModelLoader
from logger.customlogger import CustomLogger
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from Propmt.propmt_lib import PROMPT_REGISTRY
from typing import List, Dict, Union, Set
from src.datasetAnalysis.data_type_analysis import DataTypeAnalyzer
from src.datasetAnalysis.data_ingestion import datasetHandler
import numpy as np


class _SafeRegexMatch:
    def __init__(self, match_obj):
        self._match_obj = match_obj

    def __bool__(self):
        return self._match_obj is not None

    def group(self, *args):
        if self._match_obj is None:
            return ""
        try:
            return self._match_obj.group(*args)
        except Exception:
            return ""

    def groups(self, default=None):
        if self._match_obj is None:
            return tuple()
        return self._match_obj.groups(default)

    def __getattr__(self, name):
        if self._match_obj is None:
            raise AttributeError(name)
        return getattr(self._match_obj, name)


class _SafeRegex:
    def __init__(self, regex_module):
        self._regex_module = regex_module

    def search(self, pattern, string, flags=0):
        return _SafeRegexMatch(self._regex_module.search(pattern, string, flags))

    def match(self, pattern, string, flags=0):
        return _SafeRegexMatch(self._regex_module.match(pattern, string, flags))

    def fullmatch(self, pattern, string, flags=0):
        return _SafeRegexMatch(self._regex_module.fullmatch(pattern, string, flags))

    def __getattr__(self, name):
        return getattr(self._regex_module, name)


class FeatureEngineer1:
    def __init__(self, dataset: Union[str, pd.DataFrame]):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.log.info("Initializing FeatureEngineer1...")

            
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.log.info("LLM model loaded successfully")

            self.parser = JsonOutputParser(pydantic_object=ColumnRecommendation)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.propmt = PROMPT_REGISTRY['feature_engineering']

            
            if isinstance(dataset, str):
                self.df = pd.read_csv(dataset)
                self.log.info(f"Loaded dataset from path: {dataset}", dataset_shape=self.df.shape)
            elif isinstance(dataset, pd.DataFrame):
                self.df = dataset.copy()
                self.log.info("Dataset provided as DataFrame", dataset_shape=self.df.shape)
            else:
                raise AutoML_Exception("Input dataset must be a CSV path or pandas DataFrame")

            
            self.handler = datasetHandler()
            self.session_id = self.handler.save_dataset(self.df, "raw_file.csv")
            self.log.info("datasetHandler initialized and raw data saved", session_id=self.session_id)

            
            self.analyzer = DataTypeAnalyzer(self.df)
            self.recommendations = self.analyzer.analyze_data_type()
            self.log.info(
                "Data type analysis completed",
                recommendations_count=len(self.recommendations) if isinstance(self.recommendations, list) else 0
            )

            
            if isinstance(self.recommendations, dict) and "columns" in self.recommendations:
                self.recommendations = self.recommendations["columns"]
                self.log.info("Extracted 'columns' from recommendations dictionary")
            if not isinstance(self.recommendations, list):
                self.log.warning("Recommendations not in expected list format; defaulting to empty list")
                self.recommendations = []

            
            self.code_snippet = self.analyzer.generate_conversion_code(self.recommendations)
            self.converted_df = self.analyzer.apply_conversions(self.df, self.recommendations)
            if self.converted_df is None:
                self.log.warning("Converted dataframe is None; falling back to original dataframe copy")
                self.converted_df = self.df.copy()
            self.log.info("Applied conversions successfully", dataframe_overview=self.converted_df.head().to_dict())

            self.log.info("FeatureEngineer1 initialized successfully")

        except Exception as e:
            self.log.error("Error while initializing FeatureEngineer1", error=str(e))
            raise AutoML_Exception(f"Error while initializing FeatureEngineer1: {e}")

    @staticmethod
    def _extract_dataframe_column_refs(code: str) -> Set[str]:
        # Capture expressions like df['col'] or converted_df["col"] used by generated snippets.
        pattern = r"(?:df|converted_df)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
        return set(re.findall(pattern, code))

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
                    "Rate limit from LLM; retrying column processing",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    wait_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)

    def generate_features(self) -> tuple[pd.DataFrame, str]:
        self.log.info("Starting feature generation...")
        col = "unknown"
        try:
            for col_info in self.recommendations:
                col = col_info["column_name"]
                self.log.info("Processing column", column=col, suggested_dtype=col_info["suggested_dtype"])

                if col not in self.converted_df.columns:
                    self.log.warning("Column not found in dataframe; skipping", column=col)
                    continue

                try:

                
                    if col_info["suggested_dtype"] == "date":
                        self.converted_df[col] = pd.to_datetime(self.converted_df[col], errors='coerce')
                        self.converted_df[col + "_day"] = self.converted_df[col].dt.day
                        self.converted_df[col + "_month"] = self.converted_df[col].dt.month
                        self.converted_df[col + "_weekday"] = self.converted_df[col].dt.weekday
                        self.converted_df[col + "_hour"] = self.converted_df[col].dt.hour
                        self.converted_df[col + "_minute"] = self.converted_df[col].dt.minute
                        self.converted_df.drop(columns=[col], inplace=True, errors="ignore")
                        self.log.info("Extracted datetime features", column=col)

                
                    elif col_info["suggested_dtype"] == 'object':
                        meta_data = {
                            'column_name': col,
                            'current_dtype': str(self.converted_df[col].dtype),
                            'sample_value': list(self.converted_df[col].head(5).values),
                            'unique_values_count': len(self.converted_df[col].unique())
                        }
                        self.log.debug("Prepared metadata for object column", metadata=meta_data)

                    
                        chain = self.propmt | self.llm | self.fixing_parser
                        response = self._invoke_with_retry(chain, {
                            'return_instructions': self.parser.get_format_instructions(),
                            'meta_data': meta_data
                        })
                        self.log.info("LLM response received", response=response)

                    
                        if not isinstance(response, dict):
                            self.log.warning("Unexpected response format for feature engineering; skipping", column=col)
                            continue

                        if response.get('remake') == 'yes':
                            self.log.info("Remaking column using generated code", column=col)
                            raw_code = response.get('code', '')
                            if not raw_code.strip():
                                self.log.warning("Generated response missing code; skipping remake", column=col)
                                continue
                            clean_code = raw_code.replace("import pandas as pd", "").replace("import re", "")

                            referenced_cols = self._extract_dataframe_column_refs(clean_code)
                            missing_cols = sorted([c for c in referenced_cols if c not in self.converted_df.columns])
                            if missing_cols:
                                self.log.warning(
                                    "Generated code references missing columns; skipping remake",
                                    column=col,
                                    missing_columns=missing_cols,
                                )
                                continue

                            safe_re = _SafeRegex(re)
                            exec_globals = {"pd": pd, "re": safe_re, "np": np}
                            # Support generated snippets that use either `df` or `converted_df`.
                            exec_locals = {"converted_df": self.converted_df, "df": self.converted_df}
                            try:
                                exec(clean_code, exec_globals, exec_locals)
                            except Exception as exec_error:
                                self.log.warning(
                                    "Generated code execution failed; skipping remake",
                                    column=col,
                                    error=str(exec_error),
                                )
                                continue

                            # Keep dataframe reference updated if generated code reassigns one of the variables.
                            self.converted_df = exec_locals.get("converted_df", exec_locals.get("df", self.converted_df))
                            self.converted_df.drop(columns=[col], inplace=True, errors="ignore")
                            self.log.debug("Dropped original object column", column=col)
                        else:
                            self.log.info("No remake needed for column", column=col)
                    else:
                        self.log.info("Skipping column (no special handling)", column=col)

                except Exception as col_error:
                    self.log.warning("Skipping column due to processing error", column=col, error=str(col_error))
                    continue

            
            self.handler = datasetHandler(session_id=self.session_id)
            cleaned_df = self.converted_df.dropna().drop_duplicates()
            if cleaned_df.empty and not self.converted_df.empty:
                self.log.warning(
                    "dropna removed all rows; falling back to drop_duplicates only",
                    original_shape=str(self.converted_df.shape),
                )
                cleaned_df = self.converted_df.drop_duplicates()

            if cleaned_df.empty and not self.converted_df.empty:
                self.log.warning(
                    "drop_duplicates still empty unexpectedly; using original converted dataframe",
                    original_shape=str(self.converted_df.shape),
                )
                cleaned_df = self.converted_df.copy()

            self.converted_df = cleaned_df
            self.handler.save_dataset(self.converted_df, "processed_file.csv")
            self.log.info("Processed data saved successfully", path=self.handler.session_path)

            self.log.info("Feature generation completed successfully")
            return self.converted_df, self.session_id

        except Exception as e:
            self.log.error("Error while processing column", column=col, error=str(e))
            raise AutoML_Exception(f"Error while processing column {col}: {e}")


if __name__ == "__main__":
    try:
        dataset_path = r"D:\College\Project\automl\data\Data_Train.csv"
        fe = FeatureEngineer1(dataset_path)
        processed_df, session_id = fe.generate_features()

        print("Feature generation completed successfully!")
        print("Shape of final dataframe:", processed_df.shape)
        print("Preview of processed dataframe:")
        print(processed_df.head())

    except AutoML_Exception as ae:
        print("AutoML Exception:", ae)
    except Exception as e:
        print("Unexpected error:", str(e))
