import os
import tempfile
import numpy as np
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

import emotion_detect as ed
import emotion_tts as et

app = FastAPI(title="Emotion-Aware Voice Assistant", version="1.0")

OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "/tmp/tts_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    """Upload a short WAV/FLAC clip -> returns transcript + detected emotion.
    Splits speech-to-text and emotion detection out as its own endpoint so a
    client (or another service) can call just this step if it doesn't need a
    chat reply."""
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_obj = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_obj, language="en-IN")
        except sr.UnknownValueError:
            raise HTTPException(status_code=422, detail="Could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Speech API error: {e}")

        audio_data = np.frombuffer(audio_obj.get_wav_data(), dtype=np.int16).astype(np.float32)
        peak = np.max(np.abs(audio_data)) or 1.0
        audio_data = audio_data / peak

        emotion, confidence = ed.detect_emotion_with_model(audio_data, sr=audio_obj.sample_rate)

        return {"text": text, "emotion": emotion, "confidence": confidence}
    finally:
        os.remove(tmp_path)


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
        out_path = os.path.join(OUTPUT_DIR, f"{next(tempfile._get_candidate_names())}.wav")
        et.speak_response(reply, emotion=emotion, output_path=out_path, play=False)
        response["audio_url"] = f"/audio/{os.path.basename(out_path)}"

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
    out_path = os.path.join(OUTPUT_DIR, f"{next(tempfile._get_candidate_names())}.wav")
    et.speak_response(reply, emotion=emotion, output_path=out_path, play=False)
    return {"reply": reply, "audio_url": f"/audio/{os.path.basename(out_path)}"}
