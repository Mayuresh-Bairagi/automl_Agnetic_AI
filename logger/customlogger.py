import logging
import os
from datetime import datetime
from pathlib import Path

import structlog


class CustomLogger:
    """Structured JSON logger with both file and console output.

    Each ``CustomLogger`` instance writes to a timestamped log file inside
    *log_dir*.  Calling :meth:`deleteLog` keeps only the most-recent *n* log
    files, preventing unbounded disk growth.

    Parameters
    ----------
    log_dir : str
        Directory (relative to ``os.getcwd()``) where ``.log`` files are
        written.  Created automatically if it does not exist.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name: str = __file__) -> structlog.BoundLogger:
        """Return a named structlog logger wired to file + console handlers.

        Parameters
        ----------
        name : str
            Typically ``__file__`` or ``__name__`` of the calling module.
            Only the *basename* is used as the logger name.
        """
        logger_name = os.path.basename(name)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # Avoid adding duplicate handlers on repeated calls
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[file_handler, console_handler],
            )
        else:
            root_logger.setLevel(logging.INFO)
            if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
                root_logger.addHandler(file_handler)

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)

    def delete_old_logs(self, n: int = 5) -> None:
        """Remove all but the *n* most-recent log files.

        Parameters
        ----------
        n : int
            Number of log files to retain (default 5).
        """
        log_dir = Path(self.logs_dir)
        log_files = sorted(
            log_dir.glob("*.log"),
            key=lambda f: f.stat().st_mtime,
        )
        for f in log_files[:-n]:
            f.unlink()

    # Keep the old name as an alias for backward compatibility
    def deleteLog(self, n: int = 5) -> None:  # noqa: N802
        self.delete_old_logs(n)


if __name__ == "__main__":
    logger_instance = CustomLogger()
    logger = logger_instance.get_logger(__file__)
    logger.info("User uploaded a file", user_id=123, filename="report.csv")
    logger.error("Failed to process CSV", error="File not found", user_id=123)
    logger_instance.delete_old_logs()
   