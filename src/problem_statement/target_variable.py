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
    
    def get_target_variable(self,Problem_Statement):    
        try:
            column_names = list(self.df.columns)
            response = self.chain.invoke(
                {'return_instructions' : self.parser.get_format_instructions(),
                'problem_statement' : Problem_Statement,
                'columnnames' : column_names
                }
            )
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
