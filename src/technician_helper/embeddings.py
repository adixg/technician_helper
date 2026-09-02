"""Cached SentenceTransformer loader.

Loading a SentenceTransformer costs seconds and hundreds of MB of RAM. The
retrieval path used to do it on every single query; this caches one instance
per (model name, device) for the life of the process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from technician_helper.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load(model_name: str, device: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        token=settings.hf_token,
    )


def get_embedding_model(
    model_name: str | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    """Return a cached embedding model, loading it on first use."""
    return _load(model_name or settings.embed_model, device or resolve_device())
