"""Custom exception type that captures rich file/line context.

``HCException`` wraps an underlying error and records where it occurred, which is
especially helpful in a multi-stage ML pipeline (data generation/loading,
preprocessing, model fitting, explanation, serving) where a bare traceback can be
hard to localise.
"""
from __future__ import annotations

import sys
import traceback
from types import TracebackType
from typing import Optional


def _format_detail(error: BaseException, exc_tb: Optional[TracebackType]) -> str:
    """Build a readable message including file and line information."""
    if exc_tb is None:
        return f"Error: {error}"
    last = exc_tb
    while last.tb_next is not None:
        last = last.tb_next
    file_name = last.tb_frame.f_code.co_filename
    line_no = last.tb_lineno
    return (
        f"Error in [{file_name}] at line [{line_no}]: "
        f"{type(error).__name__}: {error}"
    )


class HCException(Exception):
    """Application-specific exception carrying file/line context."""

    def __init__(self, error: object, error_detail: object = sys) -> None:
        super().__init__(str(error))
        exc_tb = None
        if hasattr(error_detail, "exc_info"):
            _, _, exc_tb = error_detail.exc_info()
        base_err = error if isinstance(error, BaseException) else Exception(str(error))
        self.message = _format_detail(base_err, exc_tb)

    def __str__(self) -> str:  # noqa: D105
        return self.message


def format_exception() -> str:
    """Return the currently-handled exception as a formatted string."""
    return "".join(traceback.format_exception(*sys.exc_info()))
