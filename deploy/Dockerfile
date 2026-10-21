# Server image for Bird Sound ID (used by Render, works on any Docker host)
FROM python:3.12-slim

WORKDIR /app
COPY deploy/requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY bird_sound_analyzer.py web_app.py ./
COPY web ./web

ENV BIRD_HOSTED=1 \
    BIRD_DATA_DIR=/tmp/birdsound \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

# Render sets $PORT; default to 10000 elsewhere
CMD gunicorn --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:${PORT:-10000} web_app:app
