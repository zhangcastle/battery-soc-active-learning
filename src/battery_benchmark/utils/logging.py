"""Logging configuration for the benchmark.

Uses the standard Python ``logging`` module.  Log level is configurable.
All benchmark loggers are children of ``battery_benchmark``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure root benchmark logger.

    Parameters
    ----------
    level:
        Log level string (``"DEBUG"``, ``"INFO"``, etc.) or integer.
    log_file:
        Optional path to a log file.  Logs are always also written to stderr.

    Returns
    -------
    logging.Logger
        The ``battery_benchmark`` root logger.
    """
    logger = logging.getLogger("battery_benchmark")
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under ``battery_benchmark``."""
    return logging.getLogger(f"battery_benchmark.{name}")
