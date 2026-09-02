"""A small retry helper with exponential backoff and jitter (no external deps)."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "operation",
) -> T:
    """Call ``func`` up to ``attempts`` times, backing off between failures.

    Delay after failure *n* (1-indexed) is ``base_delay * 2**(n-1)`` seconds,
    capped at ``max_delay``, plus up to 10% jitter. Only ``exceptions`` are
    retried; anything else propagates immediately. The final failure is
    re-raised unchanged.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.1)
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
