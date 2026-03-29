import pandas as pd
import os
import html
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception

try:
    from ydata_profiling import ProfileReport
    _PROFILE_IMPORT_ERROR = None
except Exception as exc:
    ProfileReport = None
    _PROFILE_IMPORT_ERROR = exc


class EDA:
    def __init__(self, session_id: str):
        self.log = CustomLogger().get_logger(__file__)
        try:
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

    def _write_fallback_report(self, reason: str):
        """Generate a lightweight HTML EDA report when ydata-profiling is unavailable.

        This keeps the endpoint functional in constrained environments.
        """
        row_count, col_count = self.data.shape
        missing = self.data.isna().sum().to_frame("missing_count")
        missing["missing_percent"] = ((missing["missing_count"] / max(row_count, 1)) * 100).round(2)

        dtypes_df = self.data.dtypes.astype(str).to_frame("dtype")
        numeric_summary = self.data.describe(include=["number"]).T
        categorical_summary = self.data.describe(include=["object", "category", "bool"]).T
        preview = self.data.head(20)

        html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>EDA Report (Fallback)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1, h2 {{ margin-top: 24px; }}
        .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; padding: 12px; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; }}
        th {{ background: #f6f8fa; }}
        .note {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 12px; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Exploratory Data Analysis (Fallback Report)</h1>
    <p class=\"note\">
        Full profiling via ydata-profiling was unavailable. A fallback report was generated.<br/>
        <strong>Reason:</strong> {html.escape(reason)}
    </p>

    <div class=\"meta\">
        <strong>Rows:</strong> {row_count}<br/>
        <strong>Columns:</strong> {col_count}
    </div>

    <h2>Column Data Types</h2>
    {dtypes_df.to_html()}

    <h2>Missing Values</h2>
    {missing.to_html()}

    <h2>Numeric Summary</h2>
    {numeric_summary.to_html() if not numeric_summary.empty else '<p>No numeric columns found.</p>'}

    <h2>Categorical Summary</h2>
    {categorical_summary.to_html() if not categorical_summary.empty else '<p>No categorical columns found.</p>'}

    <h2>Data Preview (first 20 rows)</h2>
    {preview.to_html(index=False)}
</body>
</html>
"""

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.log.warning(
            "Fallback EDA report generated",
            output_path=self.output_path,
            reason=reason,
            rows=row_count,
            columns=col_count,
        )

        return self.output_path

    def generate_report(self):
        try:
            if ProfileReport is None:
                import_err = str(_PROFILE_IMPORT_ERROR) if _PROFILE_IMPORT_ERROR else "unknown import error"
                return self._write_fallback_report(
                    "ydata-profiling is unavailable in current environment. "
                    f"Original import error: {import_err}"
                )

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
            fallback_reason = f"Primary profiling failed: {str(e)}"
            self.log.warning(
                "Primary EDA generation failed, attempting fallback report",
                error=str(e),
                output_path=self.output_path,
            )
            try:
                return self._write_fallback_report(fallback_reason)
            except Exception as fallback_error:
                self.log.error(
                    "Error generating EDA report",
                    error=str(fallback_error),
                    output_path=self.output_path
                )
                raise AutoML_Exception("Error generating EDA report", fallback_error) from fallback_error


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
