# Use a python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Install system dependencies required for building some python packages
# and for OpenCV (libgl1, libglib2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependencies first to leverage Docker cache
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
# --frozen ensures we stick to uv.lock versions
RUN uv sync --frozen --no-install-project --no-dev

# Final stage
FROM python:3.11-slim-bookworm

# Install runtime system dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Enable the virtual environment by adding it to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Copy entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /app/

# Make it executable (if it wasn't already)
RUN chmod +x /app/docker-entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]