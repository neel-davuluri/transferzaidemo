FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying code (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py predict.py config.py paths.py ./
COPY eval/ eval/

# Copy artifacts (pre-downloaded); S3 download runs at startup if AWS creds are set
COPY artifacts/ artifacts/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
