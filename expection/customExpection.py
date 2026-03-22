
import sys
import traceback
from logger.customlogger import CustomLogger


class AutoML_Exception(Exception):
    """Custom exception for the AutoML pipeline.

    Captures the file name, line number, original error message, and full
    traceback so that errors surfaced anywhere in the pipeline carry enough
    context to be diagnosed quickly.

    Parameters
    ----------
    error_message : str | Exception
        Human-readable description of what went wrong.
    error_details : module, optional
        Pass ``sys`` (the default) so the constructor can call
        ``sys.exc_info()`` to retrieve the active exception context.  When
        called outside an ``except`` block, the traceback details will be
        unavailable and are reported as ``"N/A"``.
    """

    def __init__(self, error_message, error_details: sys = sys):
        super().__init__(str(error_message))
        self.error_message = str(error_message)

        exc_type, exc_value, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno
            self.traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.file_name = "N/A"
            self.lineno = "N/A"
            self.traceback_str = "No active exception context."

    def __str__(self) -> str:
        return (
            f"Error in [{self.file_name}] at line [{self.lineno}]\n"
            f"Message: {self.error_message}\n"
            f"Traceback: {self.traceback_str}"
        )


if __name__ == "__main__":
    try:
        a = 1 / 0
        print(a)
    except Exception as e:
        app_exc = AutoML_Exception(e)
        logger = CustomLogger()
        logger = logger.get_logger(__file__)
        logger.error(app_exc)
        raise app_exc
