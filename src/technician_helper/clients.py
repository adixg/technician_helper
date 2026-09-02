"""Shared external clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from technician_helper.config import settings

if TYPE_CHECKING:
    import weaviate


def weaviate_client() -> weaviate.WeaviateClient:
    """Open a connection to the local Weaviate instance using configured host/ports.

    The caller owns the connection and must ``.close()`` it (ideally via
    ``try/finally``).
    """
    import weaviate

    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )
