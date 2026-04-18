
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import requests
import speech_recognition as sr
import os
from emotion_tts import speak_response
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

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
        
        stats = {
            'energy': energy_mean,
            'pitch': spectral_mean,
            'zcr': zcr_mean
        }
        
        if energy_mean > 0.05 and spectral_mean > 2000:
            if zcr_mean > 0.15:
                return 'angry', 0.75, stats
            else:
                return 'happy', 0.70, stats
        elif energy_mean < 0.02 and spectral_mean < 1500:
            return 'sad', 0.65, stats
        else:
            return 'neutral', 0.60, stats
            
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return 'neutral', 0.50, {}

_cached_model = None
_cached_model_path = None

def detect_emotion_with_model(audio_data, sr=16000, model_path='E:/dlpro/emotion_model.keras', confidence_threshold=0.55):
    global _cached_model, _cached_model_path
    
    try:
        if _cached_model is None or _cached_model_path != model_path:
            print(f"Loading model from {model_path}...")
            _cached_model = keras.models.load_model(model_path)
            _cached_model_path = model_path
        
        model = _cached_model
        
        features = extract_features(audio_data, sr)
        if features is None:
            print("Feature extraction failed. Using simple detection.")
            result = detect_emotion_simple(audio_data, sr)
            return result[0], result[1]  
        
        if features.shape != (40, 100, 3):
            print(f"Warning: Unexpected feature shape {features.shape}. Expected (40, 100, 3)")
            result = detect_emotion_simple(audio_data, sr)
            return result[0], result[1]  
        
        features = np.expand_dims(features, axis=0)
        
        predictions = model.predict(features, verbose=0)
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        emotion_idx = top_3_indices[0]
        confidence = predictions[0][emotion_idx]
        
        second_emotion_idx = top_3_indices[1]
        second_confidence = predictions[0][second_emotion_idx]
        
        print(f"  Prediction breakdown:")
        for idx in top_3_indices:
            print(f"    {EMOTIONS[idx]}: {predictions[0][idx]:.2f}")
        
       
        confidence_gap = confidence - second_confidence
        if confidence_gap < 0.15: 
            print(f"Predictions too close ({confidence_gap:.2f} gap), defaulting to neutral")
            return 'neutral', float(confidence)
        
        if confidence < confidence_threshold:
            print(f"Low confidence ({confidence:.2f}), defaulting to neutral")
            return 'neutral', float(confidence)
        
        detected_emotion = EMOTIONS[emotion_idx]
        
        if detected_emotion == 'sad' and confidence < 0.75:
            print(f" Moderating {detected_emotion} (high recall, lower precision in training)")
            print(f"Confidence {confidence:.2f} < 0.75 threshold, using neutral")
            return 'neutral', float(confidence)
        
        if detected_emotion == 'fearful' and confidence < 0.70:
            print(f" Moderating {detected_emotion} prediction (confidence {confidence:.2f}), using neutral")
            return 'neutral', float(confidence)
        
        return detected_emotion, float(confidence)
        
    except FileNotFoundError:
        print(f"Warning: Model file '{model_path}' not found. Using simple detection.")
        result = detect_emotion_simple(audio_data, sr)
        return result[0], result[1] 
    except Exception as e:
        print(f"Model prediction error: {e}. Using simple detection.")
        import traceback
        traceback.print_exc()
        result = detect_emotion_simple(audio_data, sr)
        return result[0], result[1] 

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

print(" Emotion detection functions loaded!")
print("  Feature extraction: MFCC + Delta + Delta2 (3 channels)")

API_KEY = 'your api key' 
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_PATH = 'E:/dlpro/emotion_model.keras'

messages = [
    {"role": "system", "content": "You are a helpful chatbot."}
]

def check_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Trained model '{MODEL_PATH}' not found!")
        print("The system will use simple heuristic-based emotion detection.")
        print("To use the trained model, train it first.\n")
        return False
    else:
        print(f"Using trained emotion model: {MODEL_PATH}")
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            
            if model.input_shape == (None, 40, 100, 3):
                print(f" Input shape matches training configuration\n")
            else:
                print(f" Warning: Unexpected input shape!")
                print(f"Expected: (None, 40, 100, 3)")
                print(f"Got: {model.input_shape}\n")
            return True
        except Exception as e:
            print(f"  Warning: Could not load model: {e}\n")
            return False

def listen_to_speech():
    recognizer = sr.Recognizer()

    recognizer.energy_threshold = 400          
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5         
    recognizer.phrase_threshold = 0.2           
    recognizer.non_speaking_duration = 1.0     

    with sr.Microphone() as source:
        print("\nListening... (speak your FULL sentence, then stay silent for 2 seconds)")
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        
        print("Ready! Start speaking now...")
        
        try:
            audio = recognizer.listen(
                source,
                timeout=10,             
                phrase_time_limit=20    
            )
            print("Recording complete, processing...")
        except sr.WaitTimeoutError:
            print("Timeout - no speech detected")
            return None, None

    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print(f"\n You said: '{text}'")
        print(f"   (Length: {len(text.split())} words)")
        
        audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))  
        
        emotion, confidence = detect_emotion_with_model(
            audio_data, 
            sr=audio.sample_rate,
            model_path=MODEL_PATH
        )
        
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})\n")
        
        return text, emotion

    except sr.WaitTimeoutError:
        print("No speech detected")
    except sr.UnknownValueError:
        print("Could not understand audio - please speak more clearly")
    except sr.RequestError as e:
        print(f"Speech API error: {e}")

    return None, None

def get_answer(user_message, detected_emotion='neutral'):
    emotion_system_prompt = get_emotion_prompt(detected_emotion)
    
    emotion_messages = [
        {"role": "system", "content": emotion_system_prompt}
    ] + messages[1:]
    
    emotion_messages.append({
        "role": "user", 
        "content": f"{user_message} [User seems {detected_emotion}]"
    })

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Voice Assistant"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": emotion_messages
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()

            if not reply:
                reply = "I couldn't generate a response."

            messages.append({"role": "assistant", "content": reply})
            return reply

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

print("Voice assistant functions loaded!")

def listen_to_speech_alternative():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\n ALTERNATIVE MODE")
        print("Press ENTER when ready to speak...")
        input()
        
        print("RECORDING - Speak now!")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        print("   (Speaking for up to 5 seconds, or until you pause...)")
        audio = recognizer.record(source, duration=5)
        print("Recording stopped")
        
    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print(f"\n You said: '{text}'")
        
        audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        emotion, confidence = detect_emotion_with_model(
            audio_data, 
            sr=audio.sample_rate,
            model_path=MODEL_PATH
        )
        
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})\n")
        return text, emotion
        
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

print("Alternative listening mode available (use listen_to_speech_alternative if issues persist)")

print("="*60)
print("Emotion-Aware Voice Assistant")
print("="*60)

check_model_exists()
print("Say 'exit' to quit\n")

while True:
    query, emotion = listen_to_speech()

    if not query:
        continue

    if query.lower() in ["exit", "quit", "stop"]:
        print("Goodbye!")
        break

    reply = get_answer(query, emotion)
    print(f"Bot ({emotion} mode):", reply)
    speak_response(reply, emotion=emotion ,emotion_model_path='E:/dlpro/emotion_model.keras')
    print()