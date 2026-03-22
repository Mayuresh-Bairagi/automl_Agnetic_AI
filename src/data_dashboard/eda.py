import pandas as pd
from ydata_profiling import ProfileReport
import os
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception


class EDA:
    def __init__(self, session_id: str):
        try:
            self.log = CustomLogger().get_logger(__file__)
            
            self.session_id = session_id
            self.data_path = os.path.join(
                os.getcwd(), 'data', 'datasetAnalysis', session_id, 'processed_file.csv'
            )
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Processed file not found at {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            self.output_path = os.path.join(
                os.getcwd(), 'data', 'datasetAnalysis', session_id, 'index.html'
            )
            
            self.log.info(
                "EDA initialized successfully",
                session_id=session_id,
                data_path=self.data_path,
                output_path=self.output_path
            )
        
        except Exception as e:
            self.log.error(
                "Error initializing EDA",
                error=str(e),
                session_id=session_id
            )
            raise AutoML_Exception("Error initializing EDA", e) from e

    def generate_report(self):
        try:
            self.log.info("EDA report generation started", output_path=self.output_path)
            
            profile = ProfileReport(
                self.data, 
                title="Exploratory Data Analysis", 
                explorative=True
            )
            
            profile.to_file(self.output_path)
            
            self.log.info(
                "EDA report generated successfully",
                output_path=self.output_path,
                rows=self.data.shape[0],
                columns=self.data.shape[1]
            )
            return self.output_path
        
        except Exception as e:
            self.log.error(
                "Error generating EDA report",
                error=str(e),
                output_path=self.output_path
            )
            raise AutoML_Exception("Error generating EDA report", e) from e


if __name__ == "__main__":
    try:
        session_id = "session_id_20250923_184647_aee25535"
        eda = EDA(session_id=session_id)

        html_path = eda.generate_report()
        print(f"EDA report successfully generated at: {html_path}")

    except AutoML_Exception as e:
        print("AutoML Exception:", e)
    except Exception as e:
        print("Unexpected Error:", e)
