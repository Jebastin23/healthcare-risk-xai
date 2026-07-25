"""Project-wide logging utility.

Provides a single :func:`get_logger` factory that returns a configured
:class:`logging.Logger`. Logs are written both to the console (INFO+) and to a
timestamped, size-rotating file inside ``logs/`` (DEBUG+). Using one factory
everywhere guarantees a consistent format and avoids duplicate handlers.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"hc_{datetime.now():%Y%m%d}.log"

_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "hc", level: int = logging.DEBUG) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Root level for the logger. Handlers apply their own thresholds.

    Returns
    -------
    logging.Logger
        A logger with console and rotating-file handlers attached exactly once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured -> avoid duplicate handlers
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # Console handler -------------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler (5 MB x 5 backups) ------------------------------
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
