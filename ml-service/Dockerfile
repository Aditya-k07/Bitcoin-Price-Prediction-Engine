# Build stage - compile packages
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies with optimized pip configuration
# Using --no-build-isolation speeds up build, --no-cache-dir reduces warnings
RUN pip install --user --no-cache-dir --no-build-isolation -r requirements.txt && \
    rm -rf /tmp/pip-* && \
    find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Runtime stage - minimal
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY app/ ./app/
COPY data/ ./data/

# Create directory for saved models
RUN mkdir -p saved_models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
