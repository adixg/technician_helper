"""One-line logging setup, safe to call repeatedly."""

from __future__ import annotations

import logging

from technician_helper.config import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    _CONFIGURED = True
