"""Logging configuration helpers using Logfire."""

from __future__ import annotations

import logging
import os
from typing import Optional

import logfire

from .config import get_settings


_configured = False


def configure_logging(service_name: Optional[str] = "shl-recommender") -> None:
    """Configure Logfire and standard logging.

    This function is idempotent to avoid duplicate handlers during tests or
    repeated imports.
    """

    global _configured
    if _configured:
        return

    settings = get_settings()

    configure_kwargs: dict[str, str | bool] = {"service_name": service_name}

    if settings.logfire_api_key:
        os.environ["LOGFIRE_API_KEY"] = settings.logfire_api_key
        configure_kwargs["token"] = settings.logfire_api_key
    if settings.logfire_project:
        os.environ["LOGFIRE_PROJECT"] = settings.logfire_project
    configure_kwargs["send_to_logfire"] = "if-token-present"

    logfire.configure(**configure_kwargs)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    _configured = True
