# FastAPI + Gradio container for AQI Predictor

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps (lightgbm needs OpenMP runtime, curl for health checks)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Feast for feature store integration
RUN pip install --no-cache-dir feast[sqlite]

# Copy the application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app
WORKDIR /app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - updated to match your WebApp structure
CMD ["uvicorn", "WebApp.Backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


