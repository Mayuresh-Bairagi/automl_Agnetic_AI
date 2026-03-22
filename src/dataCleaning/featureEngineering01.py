import pandas as pd
import re
from expection.customExpection import AutoML_Exception
from model.models import *
from utils.model_loader import ModelLoader
from logger.customlogger import CustomLogger
from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from Propmt.propmt_lib import PROMPT_REGISTRY
from typing import List, Dict, Union
from src.datasetAnalysis.data_type_analysis import DataTypeAnalyzer
from src.datasetAnalysis.data_ingestion import datasetHandler
import numpy as np


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

            
            self.code_snippet = self.analyzer.generate_conversion_code(self.recommendations)
            self.converted_df = self.analyzer.apply_conversions(self.df, self.recommendations)
            self.log.info("Applied conversions successfully", dataframe_overview=self.converted_df.head().to_dict())

            self.log.info("FeatureEngineer1 initialized successfully")

        except Exception as e:
            self.log.error("Error while initializing FeatureEngineer1", error=str(e))
            raise AutoML_Exception(f"Error while initializing FeatureEngineer1: {e}")

    def generate_features(self) -> tuple[pd.DataFrame, str]:
        self.log.info("Starting feature generation...")
        try:
            for col_info in self.recommendations:
                col = col_info["column_name"]
                self.log.info("Processing column", column=col, suggested_dtype=col_info["suggested_dtype"])

                
                if col_info["suggested_dtype"] == "date":
                    self.converted_df[col] = pd.to_datetime(self.converted_df[col], errors='coerce')
                    self.converted_df[col + "_day"] = self.converted_df[col].dt.day
                    self.converted_df[col + "_month"] = self.converted_df[col].dt.month
                    self.converted_df[col + "_weekday"] = self.converted_df[col].dt.weekday
                    self.converted_df[col + "_hour"] = self.converted_df[col].dt.hour
                    self.converted_df[col + "_minute"] = self.converted_df[col].dt.minute
                    self.converted_df.drop(columns=[col], inplace=True)
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
                    response = chain.invoke({
                        'return_instructions': self.parser.get_format_instructions(),
                        'meta_data': meta_data
                    })
                    self.log.info("LLM response received", response=response)

                    
                    if response.get('remake') == 'yes':
                        self.log.info("Remaking column using generated code", column=col)
                        clean_code = response['code'].replace("import pandas as pd", "").replace("import re", "")
                        exec_globals = {"pd": pd, "re": re,"np" :np}
                        exec_locals = {"converted_df": self.converted_df}
                        exec(clean_code, exec_globals, exec_locals)
                        self.converted_df.drop(columns=[col], inplace=True)
                        self.log.debug("Dropped original object column", column=col)
                    else:
                        self.log.info("No remake needed for column", column=col)

                else:
                    self.log.info("Skipping column (no special handling)", column=col)

            
            self.handler = datasetHandler(session_id=self.session_id)
            self.converted_df = self.converted_df.dropna().drop_duplicates()
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
