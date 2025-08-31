import logging
import logging.config
import os
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging with console output and optional file handler.

    - level: e.g. "DEBUG", "INFO", "WARNING", "ERROR".
    - log_file: optional path to write logs (rotating daily kept for 7 days).
    """
    level = (level or "INFO").upper()

    handlers = ["console"]
    handler_defs: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "console",
            "stream": "ext://sys.stdout",
        }
    }

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append("file")
        handler_defs["file"] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": level,
            "formatter": "console",
            "filename": str(log_path),
            "when": "midnight",
            "backupCount": 7,
            "encoding": "utf-8",
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": handler_defs,
        "root": {"level": level, "handlers": handlers},
    }

    logging.config.dictConfig(config)
