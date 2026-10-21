# Hugging Face Spaces image for Bird Sound ID (runs web_app.py with gunicorn)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY --chown=user . .
USER user

ENV BIRD_HOSTED=1 \
    PORT=7860 \
    BIRD_DATA_DIR=/tmp/birdsound \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    TF_CPP_MIN_LOG_LEVEL=3

EXPOSE 7860
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:7860", "web_app:app"]
