"""Structured logging for pgw.

Configures the ``pgw`` logger with a Rich handler for terminal output and
optional file output via ``PGW_LOG_FILE``.  Call ``setup_logging()`` at
startup from the CLI callback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

PGW_LOGGER_NAME = "pgw"
_RICH_CONSOLE = Console(stderr=True)
_log_initialized = False


def setup_logging() -> None:
    """Configure the pgw logger with Rich terminal handler.

    Called automatically from ``cli.app.main()``.  Safe to call multiple
    times (no-op after first call).

    Environment variables:
        ``PGW_LOG_FILE`` — write logs to this file (default: none)
        ``PGW_LOG_LEVEL`` — set log level (default: INFO, DEBUG when
            ``PGW_DEBUG=1``)
    """
    global _log_initialized
    if _log_initialized:
        return

    logger = logging.getLogger(PGW_LOGGER_NAME)

    level_name = os.environ.get("PGW_LOG_LEVEL")
    if level_name:
        level = getattr(logging, level_name.upper(), logging.INFO)
    elif os.environ.get("PGW_DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)

    # Already configured — don't add duplicate handlers
    if logger.handlers:
        _log_initialized = True
        return

    # Rich terminal handler (stderr, no timestamp — Rich adds its own)
    rich_handler = RichHandler(
        console=_RICH_CONSOLE,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # Optional file handler
    log_file = os.environ.get("PGW_LOG_FILE")
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # always verbose to file
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _log_initialized = True


def get_logger(name: str = PGW_LOGGER_NAME) -> logging.Logger:
    """Return a child logger under the pgw namespace."""
    return logging.getLogger(name)


def set_verbose(enabled: bool) -> None:
    """Set the pgw logger to DEBUG (True) or INFO (False)."""
    logger = logging.getLogger(PGW_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    for handler in logger.handlers:
        # Only adjust the terminal handler, not the file handler
        if isinstance(handler, RichHandler):
            handler.setLevel(logger.level)


def set_quiet(enabled: bool) -> None:
    """Suppress all output below WARNING (True) or restore normal (False)."""
    logger = logging.getLogger(PGW_LOGGER_NAME)
    logger.setLevel(logging.WARNING if enabled else logging.INFO)
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(logger.level)


def log_to_file(path: str | Path) -> None:
    """Add a file handler at runtime (for long-running commands like serve)."""
    logger = logging.getLogger(PGW_LOGGER_NAME)
    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    logger.debug("Log file opened: %s", log_path)
