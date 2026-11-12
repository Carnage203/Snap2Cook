"""Logging with a per-run request id, console + rotating file output."""
from __future__ import annotations

import contextvars
import logging
import uuid
from logging.handlers import RotatingFileHandler

from snap2cook.config import Settings
from snap2cook.exceptions import FileOperationError
from snap2cook.fileio.safe_io import SafeFile

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    """Start a new request scope and return its short id."""
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    return rid


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure the ``snap2cook`` logger once and return it."""
    logger = logging.getLogger("snap2cook")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        SafeFile.ensure_dir(settings.log_dir)
        handlers.append(RotatingFileHandler(settings.log_dir / "snap2cook.log", maxBytes=500_000, backupCount=2, encoding="utf-8"))
    except (FileOperationError, OSError):
        pass  # console logging is enough if the disk is read-only
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(_RequestIdFilter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger
