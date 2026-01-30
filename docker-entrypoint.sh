#!/bin/bash
set -e

# Run migrations
echo "Running alembic migrations..."
alembic upgrade head

# Start the application
# Using uvicorn directly for now, but gunicorn could be used for higher load
echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
