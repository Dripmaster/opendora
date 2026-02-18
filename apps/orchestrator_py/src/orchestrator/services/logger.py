from __future__ import annotations

import logging
import sys

import structlog


LEVEL_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def create_logger(level: str):
    resolved = LEVEL_MAP.get(level.lower(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=resolved)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    return structlog.get_logger("opendora")
