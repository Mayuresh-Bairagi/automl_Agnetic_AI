from Propmt.propmt_lib import PROMPT_REGISTRY
from langchain_classic.output_parsers import PydanticOutputParser
from model.models import * 
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser 
from utils.model_loader import ModelLoader
import os 
import sys
from pathlib import Path
import pandas as pd
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception


def _safe_sample_values(series: pd.Series, max_items: int = 3):
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return []
    return non_null.head(max_items).tolist()

class TargetVariable:
    def __init__(self, session_id):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.dataset_path = os.path.join(
                    os.getcwd(), 'data', 'datasetAnalysis', session_id, 'processed_file.csv'
                )
            self.df = pd.read_csv(self.dataset_path)
            self.log.info("Dataset loaded successfully")
            self.target_variable = PROMPT_REGISTRY['target_variable']
            self.parser = JsonOutputParser(pydantic_object=TargetVariableRecommendation)
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser,llm =self.llm)
            self.chain = self.target_variable | self.llm | self.fixing_parser
            self.log.info("TargetVariable handler initialized successfully")
        except Exception as e:
            self.log.error('Error initializing Target Variable Handler', error=str(e))
            raise AutoML_Exception("Error initializing Target Variable Handler", e) from e

    def _build_column_profiles(self) -> dict:
        profiles = {}
        total_rows = max(1, len(self.df))
        for col in self.df.columns:
            series = self.df[col]
            profiles[str(col)] = {
                "dtype": str(series.dtype),
                "non_null_ratio": round(float(series.notna().sum() / total_rows), 4),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": _safe_sample_values(series),
            }
        return profiles

    def _validate_target_response(self, response: dict) -> None:
        if not isinstance(response, dict):
            raise ValueError("Target selector returned invalid response type")

        target = response.get("target_variable")
        problem_type = str(response.get("problem_type", "")).lower()
        valid_types = {"regression", "classification", "clustering"}

        if target not in self.df.columns:
            raise ValueError(f"Predicted target '{target}' not found in dataset columns")
        if problem_type not in valid_types:
            raise ValueError(f"Invalid problem_type '{problem_type}' returned by target selector")
    
    def get_target_variable(self,Problem_Statement):    
        try:
            column_names = list(self.df.columns)
            column_profiles = self._build_column_profiles()
            response = self.chain.invoke(
                {'return_instructions' : self.parser.get_format_instructions(),
                'problem_statement' : Problem_Statement,
                'columnnames' : column_names,
                'column_profiles': column_profiles,
                }
            )
            self._validate_target_response(response)
            self.log.info("Target variable prediction completed")
            return response,self.df
        except Exception as e :
            self.log.error('Error in getting target variable', error=str(e))
            raise AutoML_Exception("Error in getting target variable", e) from e


if __name__ == "__main__":
    import sys
    session_id =   "session_id_20250925_101824_7a7c48d7"  
    problem_statement = "Predicted the price of plane ticket"  
    try:
        target_var_handler = TargetVariable(session_id=session_id)
        result,dataframe = target_var_handler.get_target_variable(problem_statement)
        
        print("Predicted Target Variable:")
        print(result)

        print("frist five row of dataframe:")
        print(dataframe.head())

    except Exception as e:
        print(f"Error occurred: {e}")
