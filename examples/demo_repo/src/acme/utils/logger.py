"""Structured JSON logging for services."""
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger wired to stdout with the JSON formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
