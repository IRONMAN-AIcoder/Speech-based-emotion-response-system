# --- build/deps stage kept simple and single-stage on purpose: tensorflow's
# wheel is already a prebuilt binary, so there's little to gain from a
# multi-stage build here beyond extra complexity.
FROM python:3.12-slim

# System libraries the audio stack needs at runtime:
#   ffmpeg        -> librosa/audioread decoding of compressed audio
#   libsndfile1   -> soundfile (reading/writing wav)
#   libportaudio2 -> sounddevice import (playback itself is disabled in the
#                    server, but the import must not crash the process)
#   espeak-ng     -> pyttsx3's Linux TTS backend, only used if edge-tts
#                    (the network-based primary TTS path) fails
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py emotion_detect.py emotion_tts.py entrypoint.sh healthcheck.py ./
COPY static/ ./static/
RUN chmod +x entrypoint.sh

# Model weights are NOT baked into the image - see docker-compose.yml / the
# README for why. Locally they're bind-mounted; on Railway entrypoint.sh
# downloads them into the attached volume on first boot.
RUN mkdir -p /app/models /tmp/tts_out

ENV EMOTION_MODEL_PATH=/app/models/emotion_model.keras \
    TTS_WEIGHTS_DIR=/app/models \
    TTS_OUTPUT_DIR=/tmp/tts_out \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python healthcheck.py || exit 1

CMD ["./entrypoint.sh"]
