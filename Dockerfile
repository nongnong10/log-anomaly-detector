FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# Install dependencies first (leverages layer caching)
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Adjust ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

