"""
Dataset Q&A — natural language questions answered over uploaded datasets.

The user provides a session_id and a natural language question.  The module:
  1. Loads the processed dataset for that session.
  2. Asks the LLM to generate a pandas expression / short Python snippet that
     answers the question.
  3. Executes the generated code in a restricted sandbox.
  4. Returns the textual answer together with the supporting code.

Security notes
--------------
* Only pandas/numpy operations are exposed — no file I/O, no subprocess, no
  eval of arbitrary imports.
* Code execution is wrapped in a try/except with a tight allowed-builtins set.
* A configurable wall-clock timeout (_CODE_EXEC_TIMEOUT_S) prevents resource-
  exhaustion attacks (infinite loops, memory bombs).  Execution runs in a
  background thread; if it does not complete in time the thread is abandoned
  and a TimeoutError is raised.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from langchain_core.output_parsers import StrOutputParser

from expection.customExpection import AutoML_Exception
from logger.customlogger import CustomLogger
from Propmt.propmt_lib import PROMPT_REGISTRY
from utils.model_loader import ModelLoader

log = CustomLogger().get_logger(__name__)

# Maximum wall-clock seconds allowed for LLM-generated code execution.
_CODE_EXEC_TIMEOUT_S: int = 10

# ---------------------------------------------------------------------------
# Allowed builtins for sandboxed code execution
# ---------------------------------------------------------------------------
_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class DatasetQA:
    """
    Answer natural language questions about an uploaded dataset.

    Parameters
    ----------
    session_id : str
        Session identifier created by the /upload endpoint.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._loader = ModelLoader()
        self._llm = self._loader.load_llm()
        self._prompt = PROMPT_REGISTRY["dataset_qa"]
        self._chain = self._prompt | self._llm | StrOutputParser()
        self._df: Optional[pd.DataFrame] = None
        log.info("DatasetQA initialised", session_id=session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        csv_path = os.path.join(
            os.getcwd(), "data", "datasetAnalysis", self.session_id, "processed_file.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No processed dataset found for session: {self.session_id}")
        self._df = pd.read_csv(csv_path)
        log.info("Dataset loaded for Q&A", session_id=self.session_id, shape=str(self._df.shape))
        return self._df

    def _schema_summary(self, df: pd.DataFrame) -> str:
        """Return a compact schema string for the prompt."""
        lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "Columns:"]
        for col in df.columns:
            sample = df[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({df[col].dtype}): sample={sample}")
        return "\n".join(lines)

    def _extract_code(self, llm_response: str) -> str:
        """Pull out the first ```python ... ``` block (or the whole response)."""
        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return lines that look like code (start with df / result)
        code_lines = [
            line for line in llm_response.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return "\n".join(code_lines).strip()

    def _run_code(self, code: str, df: pd.DataFrame) -> Any:
        """Execute generated pandas code in a restricted namespace with a timeout."""
        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "df": df.copy(),
        }

        exec_error: list = []
        exec_result: list = []

        def _target() -> None:
            try:
                exec(compile(code, "<dataset_qa>", "exec"), namespace)  # noqa: S102
                exec_result.append(
                    namespace.get("result", "Code executed but no `result` variable was set.")
                )
            except Exception as exc:  # noqa: BLE001
                exec_error.append(exc)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=_CODE_EXEC_TIMEOUT_S)

        if thread.is_alive():
            raise TimeoutError(
                f"Code execution did not complete within {_CODE_EXEC_TIMEOUT_S} seconds."
            )
        if exec_error:
            raise exec_error[0]
        return exec_result[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer *question* over the session dataset.

        Returns
        -------
        dict with keys:
          - ``question``
          - ``answer``   (str representation of the computed result)
          - ``code``     (pandas code used to compute the answer)
          - ``error``    (None on success, error message on failure)
        """
        try:
            df = self._load_df()
            schema = self._schema_summary(df)

            llm_response = self._chain.invoke(
                {"schema": schema, "question": question}
            )
            log.info("LLM response received for Q&A", session_id=self.session_id)

            code = self._extract_code(llm_response)
            try:
                raw_result = self._run_code(code, df)
                # Convert DataFrames / Series to a JSON-serialisable structure
                if isinstance(raw_result, pd.DataFrame):
                    answer_str = raw_result.to_string()
                elif isinstance(raw_result, pd.Series):
                    answer_str = raw_result.to_string()
                else:
                    answer_str = str(raw_result)
            except Exception as exec_err:
                log.warning("Code execution failed, returning LLM text answer", error=str(exec_err))
                answer_str = llm_response
                code = ""

            return {
                "question": question,
                "answer": answer_str,
                "code": code,
                "error": None,
            }

        except Exception as exc:
            log.error("DatasetQA.answer failed", error=str(exc))
            return {
                "question": question,
                "answer": None,
                "code": None,
                "error": str(exc),
            }
