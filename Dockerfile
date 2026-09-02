FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the package (dependency resolution happens here)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Runtime data (source manuals + incident CSV)
COPY data ./data

# Drop privileges
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "src/technician_helper/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
