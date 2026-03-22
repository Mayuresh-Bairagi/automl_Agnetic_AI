import pandas as pd
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

            response = chain.invoke(
                {
                    'return_instructions': self.parser.get_format_instructions(),
                    'Column_metadata': column_meta_data
                }
            )

            self.log.info("Metadata extraction successful", keys=response)
            return response

        except Exception as e:
            self.log.error(f"Error during data type analysis: {e}")
            raise AutoML_Exception(f"Error during data type analysis: {e}")

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
                if suggested == "date":
                    df[name] = pd.to_datetime(df[name], errors="coerce", dayfirst=True)
                elif suggested == "time":
                    df[name] = pd.to_datetime(df[name], errors="coerce").dt.time
                elif suggested in ["integer", "int"]:
                    df[name] = df[name].astype(int)
                elif suggested == "float":
                    df[name] = df[name].astype(float)
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

