#!/bin/sh
set -e

MODELS_DIR="${TTS_WEIGHTS_DIR:-/app/models}"
mkdir -p "$MODELS_DIR"

# Each of these env vars should be a direct-download URL (Hugging Face Hub
# "resolve/main/..." links work well and don't need auth for public repos).
# If a var is unset, that file is skipped - useful if you've already put
# weights on the volume by hand once and don't want to re-download them.
download_if_missing () {
  url="$1"
  dest="$2"
  if [ -n "$url" ] && [ ! -f "$dest" ]; then
    echo "Downloading $(basename "$dest")..."
    curl -fL --retry 3 -o "$dest" "$url"
  elif [ -f "$dest" ]; then
    echo "$(basename "$dest") already present, skipping download."
  else
    echo "WARNING: no URL set and $(basename "$dest") not found on disk."
  fi
}

download_if_missing "$EMOTION_MODEL_URL" "$MODELS_DIR/emotion_model.keras"
download_if_missing "$TTS_CONDITIONER_URL" "$MODELS_DIR/emotion_tts_gan_conditioner_weights.h5"
download_if_missing "$TTS_VOCODER_URL" "$MODELS_DIR/emotion_tts_gan_vocoder_weights.h5"

# Railway (and most PaaS platforms) inject the port to listen on via $PORT
# rather than letting you hardcode it - fall back to 8000 for local/compose runs.
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
