"""Centralised configuration.

All tunable values (Weaviate connection, collection names, model ids, data paths)
live here and can be overridden with environment variables or a local ``.env`` file.
Import the module-level ``settings`` object rather than reading ``os.environ``
directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials -------------------------------------------------------
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face access token, used for embeddings and the Inference API.",
    )

    # --- Weaviate --------------------------------------------------------
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # --- Collections -----------------------------------------------------
    manual_collection: str = "ManualChunk"
    incident_collection: str = "IncidentLogs"

    # --- Models --------------------------------------------------------
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Reliability ---------------------------------------------------
    llm_timeout: float = 60.0
    llm_max_attempts: int = 3  # network retries per LLM call
    llm_repair_attempts: int = 2  # re-asks when the model breaks the schema
    weaviate_connect_attempts: int = 5
    weaviate_connect_backoff: float = 1.0

    # --- Observability -----------------------------------------------
    log_level: str = "INFO"

    # --- Data layout ---------------------------------------------------
    data_dir: Path = Path("data")

    @property
    def manuals_dir(self) -> Path:
        return self.data_dir / "manuals"

    @property
    def manuals_converted_dir(self) -> Path:
        return self.data_dir / "manuals_converted"

    @property
    def manuals_sections_dir(self) -> Path:
        return self.data_dir / "manuals_sections"

    @property
    def manuals_chunks_dir(self) -> Path:
        return self.data_dir / "manuals_chunks"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def require(self, *, weaviate: bool = False) -> None:
        """Fail fast on missing runtime prerequisites.

        Always checks credentials; also probes Weaviate when ``weaviate=True``.
        Raises ``RuntimeError`` listing every problem found.
        """
        problems: list[str] = []
        if not self.hf_token:
            problems.append("HF_TOKEN is not set (needed for embeddings and the LLM).")
        if weaviate:
            from technician_helper.clients import weaviate_ready

            if not weaviate_ready():
                problems.append(
                    f"Weaviate is not reachable at {self.weaviate_host}:{self.weaviate_http_port}."
                )
        if problems:
            raise RuntimeError("Configuration problem(s):\n  - " + "\n  - ".join(problems))


settings = Settings()
