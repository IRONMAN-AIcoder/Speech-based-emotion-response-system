import os
import numpy as np
import librosa
from tensorflow import keras
import requests

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

MODEL_PATH = os.environ.get("EMOTION_MODEL_PATH", "/app/models/emotion_model.keras")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-3.5-turbo")

_cached_model = None
_cached_model_path = None


def extract_features(audio_data, sr=16000):
    try:
        audio_data, _ = librosa.effects.trim(audio_data)
        audio_data = librosa.util.fix_length(audio_data, size=sr * 3)

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        features = np.stack([mfccs, delta_mfccs, delta2_mfccs], axis=-1)
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)

        target_length = 100
        if features.shape[1] < target_length:
            features = np.pad(features, ((0, 0), (0, target_length - features.shape[1]), (0, 0)), mode='constant')
        else:
            features = features[:, :target_length, :]

        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def detect_emotion_simple(audio_data, sr=16000):
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        energy = librosa.feature.rms(y=audio_data)

        mfcc_mean = np.mean(mfccs)
        spectral_mean = np.mean(spectral_centroid)
        zcr_mean = np.mean(zcr)
        energy_mean = np.mean(energy)

        if energy_mean > 0.05 and spectral_mean > 2000:
            if zcr_mean > 0.15:
                return 'angry', 0.75
            return 'happy', 0.70
        elif energy_mean < 0.02 and spectral_mean < 1500:
            return 'sad', 0.65
        return 'neutral', 0.60
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return 'neutral', 0.50


def get_model():
    global _cached_model, _cached_model_path
    if _cached_model is None or _cached_model_path != MODEL_PATH:
        print(f"Loading model from {MODEL_PATH}...")
        _cached_model = keras.models.load_model(MODEL_PATH)
        _cached_model_path = MODEL_PATH
    return _cached_model


def detect_emotion_with_model(audio_data, sr=16000, confidence_threshold=0.55):
    try:
        model = get_model()

        features = extract_features(audio_data, sr)
        if features is None or features.shape != (40, 100, 3):
            return detect_emotion_simple(audio_data, sr)

        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features, verbose=0)

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        emotion_idx = top_3_indices[0]
        confidence = predictions[0][emotion_idx]
        second_confidence = predictions[0][top_3_indices[1]]

        if confidence - second_confidence < 0.15:
            return 'neutral', float(confidence)
        if confidence < confidence_threshold:
            return 'neutral', float(confidence)

        detected_emotion = EMOTIONS[emotion_idx]

        if detected_emotion == 'sad' and confidence < 0.75:
            return 'neutral', float(confidence)
        if detected_emotion == 'fearful' and confidence < 0.70:
            return 'neutral', float(confidence)

        return detected_emotion, float(confidence)

    except FileNotFoundError:
        print(f"Model file '{MODEL_PATH}' not found. Using simple detection.")
        return detect_emotion_simple(audio_data, sr)
    except Exception as e:
        print(f"Model prediction error: {e}. Using simple detection.")
        return detect_emotion_simple(audio_data, sr)


def get_emotion_prompt(emotion):
    emotion_prompts = {
        'happy': "You are a cheerful and enthusiastic chatbot. Respond with positive energy and excitement.",
        'sad': "You are an empathetic and supportive chatbot. Respond with compassion and understanding.",
        'angry': "You are a calm and patient chatbot. Respond in a soothing manner to help de-escalate.",
        'fearful': "You are a reassuring and comforting chatbot. Respond with gentle encouragement.",
        'neutral': "You are a helpful and professional chatbot.",
        'surprised': "You are an engaging and curious chatbot. Match their energy with interest.",
        'disgust': "You are a diplomatic and understanding chatbot. Respond with tact."
    }
    return emotion_prompts.get(emotion, emotion_prompts['neutral'])


def get_answer(user_message, detected_emotion='neutral', history=None):
    """history: optional list of prior {"role", "content"} turns supplied by the caller
    (the client is responsible for holding conversation state across requests -
    a server process should stay stateless between calls)."""
    emotion_system_prompt = get_emotion_prompt(detected_emotion)
    emotion_messages = [{"role": "system", "content": emotion_system_prompt}]
    if history:
        emotion_messages += history
    emotion_messages.append({
        "role": "user",
        "content": f"{user_message} [User seems {detected_emotion}]"
    })

    if not API_KEY:
        return "Server is missing OPENROUTER_API_KEY."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Voice Assistant"
    }
    payload = {"model": LLM_MODEL, "messages": emotion_messages}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()
            return reply or "I couldn't generate a response."
        elif response.status_code == 401:
            return "Authentication failed. Check API key."
        elif response.status_code == 402:
            return "No credits available."
        elif response.status_code == 429:
            return "Rate limit exceeded. Try again later."
        else:
            return f"API error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"
