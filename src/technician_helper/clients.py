"""Shared external clients.

The Weaviate connection is process-wide: it is opened once (with retry), reused
across calls, and closed at interpreter exit. Callers must not close it.
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING

from technician_helper.config import settings
from technician_helper.retry import retry_call

if TYPE_CHECKING:
    import weaviate

log = logging.getLogger(__name__)

_client: weaviate.WeaviateClient | None = None
_atexit_registered = False


def _connect() -> weaviate.WeaviateClient:
    import weaviate

    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    if not client.is_ready():
        client.close()
        raise RuntimeError(
            f"Weaviate at {settings.weaviate_host}:{settings.weaviate_http_port} is not ready"
        )
    return client


def _get_or_connect(attempts: int) -> weaviate.WeaviateClient:
    global _client, _atexit_registered

    if _client is not None and _client.is_connected():
        return _client

    _client = retry_call(
        _connect,
        attempts=attempts,
        base_delay=settings.weaviate_connect_backoff,
        description="Weaviate connect",
    )
    if not _atexit_registered:
        atexit.register(close_weaviate_client)
        _atexit_registered = True
    return _client


def weaviate_client() -> weaviate.WeaviateClient:
    """Return the shared Weaviate client, connecting with retry on first use."""
    return _get_or_connect(settings.weaviate_connect_attempts)


def weaviate_ready() -> bool:
    """Cheap liveness probe: one connect attempt, never raises."""
    try:
        return _get_or_connect(1).is_ready()
    except Exception as exc:
        log.warning("Weaviate not reachable: %s", exc)
        return False


def close_weaviate_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
