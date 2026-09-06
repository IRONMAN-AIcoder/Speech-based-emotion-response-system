import os
import tempfile
import asyncio
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

import emotion_detect as ed
import emotion_tts as et

app = FastAPI(title="Emotion-Aware Voice Assistant", version="1.0")

OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "/tmp/tts_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    """Serves the browser mic client at the service's own root, so it shares
    the same HTTPS origin as the API - required for getUserMedia() mic access
    to work at all outside of localhost."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def synthesize_reply(reply: str, emotion: str) -> str:
    """Runs speak_response() in a worker thread instead of directly on the
    request coroutine. speak_response() internally does asyncio.run(...) for
    the edge-tts call, which raises RuntimeError if it's invoked on a thread
    that already has an event loop running (as FastAPI's async endpoints do).
    A plain worker thread has no event loop of its own, so asyncio.run()
    works there. Raises RuntimeError if synthesis didn't actually produce a
    file, instead of returning a URL that 404s."""
    out_path = os.path.join(OUTPUT_DIR, f"{next(tempfile._get_candidate_names())}.wav")
    ok = await asyncio.to_thread(et.speak_response, reply, emotion, ed.MODEL_PATH, out_path, False)
    if not ok or not os.path.exists(out_path):
        raise RuntimeError("speech synthesis failed - see server logs for the underlying TTS error")
    return f"/audio/{os.path.basename(out_path)}"


@app.get("/health")
def health():
    """Cheap liveness check. Does NOT load the model, so it stays fast for
    container orchestrators (k8s liveness/readiness probes, ECS health checks)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness check: actually loads the emotion model once, so orchestrators
    can wait until inference is truly usable before sending traffic."""
    try:
        ed.get_model()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"model not ready: {e}")


@app.post("/transcribe-and-detect")
async def transcribe_and_detect(audio: UploadFile = File(...)):
    """Upload a short audio clip -> returns transcript + detected emotion.
    Accepts whatever format the browser's MediaRecorder produces (webm/ogg/
    mp4/wav) via librosa+ffmpeg, not just WAV/FLAC - and standardizes
    everything to 16kHz mono before both speech recognition and emotion
    detection, since the emotion model's features assume that rate
    regardless of what the source file was recorded at."""
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        raw_path = tmp.name

    wav_path = raw_path + ".std.wav"
    try:
        try:
            audio_data, _ = librosa.load(raw_path, sr=16000, mono=True)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

        sf.write(wav_path, audio_data, 16000)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_obj = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_obj, language="en-IN")
        except sr.UnknownValueError:
            raise HTTPException(status_code=422, detail="Could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Speech API error: {e}")

        peak = np.max(np.abs(audio_data)) or 1.0
        audio_data = audio_data / peak

        emotion, confidence = ed.detect_emotion_with_model(audio_data, sr=16000)

        return {"text": text, "emotion": emotion, "confidence": confidence}
    finally:
        for p in (raw_path, wav_path):
            if os.path.exists(p):
                os.remove(p)


@app.post("/chat")
async def chat(audio: UploadFile = File(...), synthesize_speech: bool = Form(True)):
    """End-to-end: audio in -> transcript + emotion + LLM reply (+ optional
    synthesized speech file). This mirrors the original main.py loop but as a
    single stateless request instead of a blocking mic/speaker session."""
    result = await transcribe_and_detect(audio)
    text, emotion = result["text"], result["emotion"]
    reply = ed.get_answer(text, detected_emotion=emotion)

    response = {"transcript": text, "emotion": emotion, "reply": reply}

    if synthesize_speech:
        try:
            response["audio_url"] = await synthesize_reply(reply, emotion)
        except RuntimeError as e:
            response["audio_error"] = str(e)

    return response


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/reply-text-only")
async def reply_text_only(text: str = Form(...), emotion: str = Form("neutral")):
    """Skip STT entirely - useful for testing the LLM + TTS path without recording audio."""
    reply = ed.get_answer(text, detected_emotion=emotion)
    try:
        audio_url = await synthesize_reply(reply, emotion)
        return {"reply": reply, "audio_url": audio_url}
    except RuntimeError as e:
        return {"reply": reply, "audio_error": str(e)}
