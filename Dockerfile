# syntax=docker/dockerfile:1

# ================================
# Base stage - common dependencies
# ================================
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# ================================
# Builder stage - install dependencies
# ================================
FROM base as builder

# Install pip-tools for reproducible builds
RUN pip install --no-cache-dir pip-tools

# Copy dependency files
COPY requirements.in requirements.in

# Compile requirements.txt from requirements.in
RUN pip-compile --quiet requirements.in

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install --no-deps --no-cache-dir -r requirements.txt

# ================================
# Runtime stage - final production image
# ================================
FROM base as runtime

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Create cache directory
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
