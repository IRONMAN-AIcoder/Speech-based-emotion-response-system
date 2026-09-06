# Emotion-Aware Voice Assistant — Containerized Service

## What changed from your original code, and why

Your `main.py` / `emotion_tts.py` were written as an **interactive desktop
app**: it blocks on `sr.Microphone()` waiting for you to talk, and blocks on
`sd.play()` to push audio out of your speakers. That model doesn't translate
to "host it in a container" as-is — a server has no mic or speakers, and a
process that blocks forever waiting for local hardware input isn't something
an orchestrator (Docker/K8s/ECS) can health-check or scale. So the pipeline
was split into **stateless HTTP endpoints** (`app.py`) that take an uploaded
audio file in and hand a JSON reply + a synthesized `.wav` back — the same
logic (`extract_features`, `detect_emotion_with_model`, `get_answer`,
`EmotionTTSEngine`), just triggered by a request instead of a `while True`
loop on your local device.

Three other things were fixed along the way:

1. **Hardcoded Windows paths** (`E:/dlpro/emotion_model.keras`) →
   environment variables (`EMOTION_MODEL_PATH`, `TTS_WEIGHTS_DIR`) defaulting
   to `/app/models/...` inside the container.
2. **Hardcoded API key** in source (`API_KEY = 'your api key'`) → read from
   the `OPENROUTER_API_KEY` environment variable. Never bake secrets into an
   image or commit them — anyone with the image can extract them.
3. **Model weights are mounted, not baked into the image.** Your `.h5`/`.keras`
   files change every time you retrain; if they're `COPY`'d into the image,
   every retrain means a full rebuild and re-push of a multi-GB image. Mounting
   them as a volume (or pulling from S3/GCS/a model registry at container
   start) decouples "ship new code" from "ship a new model" — which is the
   whole point of MLOps versioning your model and code independently.

## Files

- `Dockerfile` — the image definition
- `requirements.txt` — pinned Python deps
- `app.py` — FastAPI service (endpoints below)
- `emotion_detect.py` — your detection + LLM-call logic, minus the mic loop
- `emotion_tts.py` — your TTS engine, with playback made optional so it
  doesn't require a working audio *output* device in the container
- `docker-compose.yml` — local run config with the model volume + env file

## Endpoints

| Method | Path                     | What it does                                              |
|--------|--------------------------|-------------------------------------------------------------|
| GET    | `/health`                | Fast liveness probe (no model load)                        |
| GET    | `/ready`                 | Loads the emotion model once; use as a readiness probe     |
| POST   | `/transcribe-and-detect` | audio file → `{text, emotion, confidence}`                  |
| POST   | `/chat`                  | audio file → transcript + emotion + LLM reply + `audio_url`|
| GET    | `/audio/{filename}`      | fetch a synthesized reply `.wav`                            |
| POST   | `/reply-text-only`       | text + emotion (form fields) → reply + `audio_url`, skips STT — handy for testing |

## Build & run locally

```bash
# 1. put your trained weights where the compose file expects them
mkdir -p models
cp /path/to/emotion_model.keras models/
cp /path/to/emotion_tts_gan_conditioner_weights.h5 models/
cp /path/to/emotion_tts_gan_vocoder_weights.h5 models/

# 2. set secrets in a .env file (docker-compose reads this automatically)
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# 3. build and run
docker compose up --build
```

Then test it:

```bash
curl -X POST http://localhost:8000/reply-text-only \
  -F "text=I just got some really great news!" \
  -F "emotion=happy"

curl -X POST http://localhost:8000/chat \
  -F "audio=@sample.wav"
```

## Building/pushing the image directly (no compose)

```bash
docker build -t your-registry/emotion-voice-assistant:v1 .
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=sk-or-... \
  -v $(pwd)/models:/app/models:ro \
  your-registry/emotion-voice-assistant:v1

docker push your-registry/emotion-voice-assistant:v1
```

## Notes for the MLOps side specifically

- **Conversation state**: the original script kept a global `messages` list
  across turns. A server handling concurrent requests can't share mutable
  global state safely across clients, so `get_answer()` now takes an
  optional `history` list — the *caller* holds conversation state (session
  store, Redis, whatever your architecture uses), and the container stays
  stateless. That's also what lets you run more than one replica behind a
  load balancer.
- **Image size**: `tensorflow-cpu` + `librosa` + `ffmpeg` will land you a
  ~2-3GB image. If you don't need GPU inference, `tensorflow-cpu` (already
  in `requirements.txt`) is meaningfully smaller than the full `tensorflow`
  wheel. If you do need GPU, switch the base image to an
  `nvidia/cuda`-based Python image and install regular `tensorflow` with
  matching CUDA/cuDNN versions.
- **Cold start**: the emotion model and TTS engine lazy-load on first use
  (matching your original `_cached_model` pattern). For a real deployment,
  call `/ready` from your orchestrator's readiness probe (not `/health`) so
  traffic isn't routed to a replica before the model is actually loaded —
  and consider warming it in a startup hook if cold-start latency matters.
- **External dependencies at runtime**: `recognize_google` (STT) and
  `edge-tts` both call out over the network. Make sure your container's
  network policy/egress rules allow that, and treat their downtime as a
  failure mode you handle (the code already falls back from the trained
  model to heuristic detection — no equivalent fallback exists for the STT
  or TTS network calls, so consider what you want to return on `RequestError`
  in production, e.g. a 502 the client can retry).
- **Secrets**: use your platform's secret manager (K8s Secrets, AWS Secrets
  Manager/SSM, Docker Swarm secrets) to inject `OPENROUTER_API_KEY` rather
  than plain environment variables in a checked-in compose file, for
  anything beyond local testing.

## Deploying to Railway

Railway builds straight from your Dockerfile and gives you a permanent
public HTTPS URL. Two files here are Railway-specific: `entrypoint.sh`
(downloads model weights into a persistent volume on first boot, since you
can't bind-mount a local `models/` folder like you can with compose) and
`railway.toml` (tells Railway to poll `/health`).

1. **Push this folder to a GitHub repo.** Railway deploys from a connected
   repo, not by uploading a zip.
2. **Host your three weight files somewhere with a direct-download URL.**
   The [Hugging Face Hub](https://huggingface.co/new) is the easiest free
   option for model weights — create a model repo, upload
   `emotion_model.keras`, `emotion_tts_gan_conditioner_weights.h5`, and
   `emotion_tts_gan_vocoder_weights.h5`, and each file's URL will look like
   `https://huggingface.co/<you>/<repo>/resolve/main/<file>`.
3. **Create the Railway project**: railway.app → New Project → Deploy from
   GitHub repo → pick this repo. Railway detects the Dockerfile automatically.
4. **Add a volume**: in the service's Settings → Volumes, add a volume
   mounted at `/app/models`. This is what makes the downloaded weights
   persist across restarts/redeploys instead of re-downloading every boot.
5. **Set environment variables** in the Variables tab (same names as
   `.env.example`): `OPENROUTER_API_KEY`, and `EMOTION_MODEL_URL`,
   `TTS_CONDITIONER_URL`, `TTS_VOCODER_URL` pointing at the files from step 2.
6. **Deploy.** Watch the build logs, then the deploy logs — you should see
   `entrypoint.sh` download each file once, then uvicorn start up. Railway
   assigns a domain automatically (Settings → Networking → Generate Domain),
   something like `https://your-service.up.railway.app`.
7. **Test it** exactly like you did locally, just with that domain instead
   of `localhost:8000`:
   ```bash
   curl https://your-service.up.railway.app/health
   curl -X POST https://your-service.up.railway.app/reply-text-only \
     -F "text=I just got some great news!" -F "emotion=happy"
   ```

That URL is what you share/use from anywhere — phone, another laptop,
wherever — since it's a real public HTTPS endpoint, not something tied to
your machine being on.

## Browser client (mic in, voice out)

The service serves a simple web client at its own root (`/`) — open the
deployed URL directly in a browser (phone or laptop) and you get a "press the
orb, say something" interface: it records your mic, sends it to `/chat`,
shows the transcript + detected emotion, and plays back the synthesized
reply automatically. The orb's glow color shifts with the detected emotion.

It's served from the same origin as the API on purpose — browsers require a
secure context (HTTPS, or `localhost`) for microphone access, and hosting it
here means it automatically satisfies that once deployed on Railway's HTTPS
domain, with no separate hosting or CORS setup needed.

To test locally: `docker compose up`, then open `http://localhost:8000` in
your browser (not `curl` — you need an actual browser tab for mic access).
`http://localhost` counts as a secure context even without HTTPS, so this
works before you deploy anything.

The "Server settings" field at the bottom of the page is only needed if you
ever host this HTML file somewhere *other* than this service itself (e.g. a
separate static host) — point it at the deployed API's full URL. Leave it
blank when using the built-in `/` route.
