"""Logger factory with console + optional rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s — %(message)s"


def get_logger(
    name: str = "counting",
    *,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    backup_count: int = 7,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(_FMT)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(p, when="midnight", backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
