import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import sounddevice as sd
import soundfile as sf
import asyncio
import os
import tempfile
from scipy import signal


EMOTION_VOICE_PROFILES = {
    'neutral': {
        'rate': '+0%', 'pitch': '+0Hz', 'volume': '+0%',
        'energy': 1.00, 'pitch_steps': 0.0, 'speed': 1.00,
        'pitch_variance': 0.0, 'emphasis_pattern': 'flat'
    },
    'happy': {
        'rate': '+15%', 'pitch': '+8Hz', 'volume': '+10%',
        'energy': 1.20, 'pitch_steps': 2.5, 'speed': 1.15,
        'pitch_variance': 0.12, 'emphasis_pattern': 'upward'
    },
    'sad': {
        'rate': '-18%', 'pitch': '-8Hz', 'volume': '-15%',
        'energy': 0.70, 'pitch_steps': -2.5, 'speed': 0.82,
        'pitch_variance': 0.06, 'emphasis_pattern': 'downward'
    },
    'angry': {
        'rate': '+12%', 'pitch': '+3Hz', 'volume': '+20%',
        'energy': 1.35, 'pitch_steps': 1.0, 'speed': 1.12,
        'pitch_variance': 0.15, 'emphasis_pattern': 'sharp'
    },
    'fearful': {
        'rate': '+10%', 'pitch': '+10Hz', 'volume': '-5%',
        'energy': 0.85, 'pitch_steps': 3.0, 'speed': 1.10,
        'pitch_variance': 0.18, 'emphasis_pattern': 'wavering'
    },
    'disgust': {
        'rate': '-10%', 'pitch': '-5Hz', 'volume': '-10%',
        'energy': 0.85, 'pitch_steps': -1.5, 'speed': 0.90,
        'pitch_variance': 0.08, 'emphasis_pattern': 'nasal'
    },
    'surprised': {
        'rate': '+20%', 'pitch': '+12Hz', 'volume': '+15%',
        'energy': 1.25, 'pitch_steps': 4.0, 'speed': 1.20,
        'pitch_variance': 0.20, 'emphasis_pattern': 'sudden'
    },
}

EMOTION_MAP = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
    'fearful': 4, 'disgust': 5, 'surprised': 6
}

NUM_EMOTIONS  = len(EMOTION_MAP)
MEL_CHANNELS  = 80
SAMPLE_RATE   = 22050
VOICE         = "en-US-JennyNeural"



def extract_emotion_embeddings(emotion_model_path='E:/dlpro/emotion_model.keras'):
    try:
        print(f"  Loading your trained model: {emotion_model_path}")
        emotion_model = keras.models.load_model(emotion_model_path)
        
        dummy = np.zeros((1, 40, 100, 3), dtype=np.float32)
        _ = emotion_model(dummy, training=False)
        print(f"Model built successfully")
        
        output_layer = None
        for layer in reversed(emotion_model.layers):
            if isinstance(layer, keras.layers.Dense) and layer.units == NUM_EMOTIONS:
                output_layer = layer
                break
        
        if output_layer is None:
            raise ValueError("Could not find final Dense(7) layer")
        
        weights = output_layer.get_weights()[0]
        print(f"Found output layer weights: {weights.shape}")
        
        emotion_embeddings = weights.T
        norms = np.linalg.norm(emotion_embeddings, axis=1, keepdims=True)
        emotion_embeddings = emotion_embeddings / (norms + 1e-8)
        
        embed_dim = emotion_embeddings.shape[1]
        print(f" Emotion embeddings extracted: shape {emotion_embeddings.shape}")
        
        return emotion_embeddings, embed_dim
    
    except Exception as e:
        print(f"Could not extract embeddings: {e}")
        return None, 256


def build_emotion_conditioner(emotion_embed_dim=128):
    mel_input   = keras.Input(shape=(MEL_CHANNELS, None), name='mel_input')
    embed_input = keras.Input(shape=(emotion_embed_dim,),  name='embed_input')
    
    params = keras.layers.Dense(MEL_CHANNELS * 2, activation='tanh')(embed_input)
    scale = keras.ops.expand_dims(params[:, :MEL_CHANNELS], axis=-1)
    bias  = keras.ops.expand_dims(params[:,  MEL_CHANNELS:], axis=-1)
    
    conditioned = mel_input * (1.0 + scale) + bias
    
    return keras.Model(
        inputs=[mel_input, embed_input],
        outputs=conditioned,
        name='EmotionConditioner'
    )


def build_vocoder_generator():
    mel_input = keras.Input(shape=(MEL_CHANNELS, None), name='mel_input')
    
    x = keras.layers.Lambda(lambda t: keras.ops.transpose(t, [0, 2, 1]))(mel_input)
    x = keras.layers.Conv1D(128, 7, padding='same')(x)
    
    channels = 128
    for stride, ksize in [(8, 16), (8, 16), (2, 4), (2, 4)]:
        x = keras.layers.LeakyReLU(0.1)(x)
        x = keras.layers.Conv1DTranspose(channels // 2, ksize, strides=stride, padding='same')(x)
        channels //= 2
        
        branches = []
        for kernel in [3, 7, 11]:
            r = x
            for dilation in [1, 3, 5]:
                r = keras.layers.LeakyReLU(0.1)(r)
                r = keras.layers.Conv1D(channels, kernel, padding='causal', dilation_rate=dilation)(r)
                r = r + x
            branches.append(r)
        x = keras.layers.Average()(branches)
    
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Conv1D(1, 7, padding='same', activation='tanh')(x)
    x = keras.layers.Lambda(lambda t: keras.ops.squeeze(t, axis=-1))(x)
    
    return keras.Model(inputs=mel_input, outputs=x, name='HiFiGAN_Generator')



def apply_smooth_pitch_shift(audio, sr, pitch_steps):
    if abs(pitch_steps) < 0.1:
        return audio

    pitch_steps = np.clip(pitch_steps, -4.0, 5.0)
    
    try:
        audio = librosa.effects.pitch_shift(
            audio.astype(np.float32),
            sr=sr,
            n_steps=pitch_steps,
            bins_per_octave=12  
        )
    except Exception as e:
        print(f"  Pitch shift skipped: {e}")
    
    return audio


def apply_emotion_prosody(audio, sr, emotion, variance):

    if variance < 0.01:
        return audio
    
    try:
        duration = len(audio) / sr
        
        t = np.arange(len(audio)) / sr
        
        if emotion == 'happy':
            prosody = 1.0 + variance * 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
            
        elif emotion == 'sad':
            prosody = 1.0 - variance * 0.10 * (t / duration)
            
        elif emotion == 'angry':
            prosody = 1.0 + variance * 0.20 * np.sin(2 * np.pi * 5 * t + np.random.randn(len(t)) * 0.3)
            
        elif emotion == 'fearful':
            prosody = 1.0 + variance * 0.15 * np.sin(2 * np.pi * 6 * t)
            
        elif emotion == 'surprised':
            rise = np.exp(-3 * t)
            prosody = 1.0 + variance * 0.25 * rise
            
        elif emotion == 'disgust':
            prosody = 1.0 + variance * 0.08 * np.sin(2 * np.pi * 1.5 * t)
            
        else: 
            prosody = np.ones(len(audio))
        
        if len(prosody) > 100:
            window_size = min(int(sr * 0.02), len(prosody) // 10)  
            if window_size > 2:
                prosody = signal.savgol_filter(prosody, window_size | 1, 2) 
        
        audio = audio * prosody
        
    except Exception as e:
        print(f"  Prosody modulation skipped: {e}")
    
    return audio


def apply_timing_variation(audio, sr, speed_factor):

    if abs(speed_factor - 1.0) < 0.02:
        return audio
    
    speed_factor = np.clip(speed_factor, 0.80, 1.25)
    
    try:
        audio = librosa.effects.time_stretch(
            audio.astype(np.float32),
            rate=speed_factor
        )
    except Exception as e:
        print(f"  Timing variation skipped: {e}")
    
    return audio


def apply_emphasis_pattern(audio, sr, emotion, emphasis_pattern):
    duration = len(audio) / sr
    t = np.arange(len(audio)) / sr
    
    try:
        if emphasis_pattern == 'upward': 
            emphasis = 1.0 + 0.15 * np.linspace(0, 1, len(audio))
            
        elif emphasis_pattern == 'downward':  
            emphasis = 1.0 - 0.20 * np.linspace(0, 1, len(audio))
            
        elif emphasis_pattern == 'sharp': 
            emphasis = np.ones(len(audio)) * 1.15
            
        elif emphasis_pattern == 'wavering': 
            emphasis = 1.0 + 0.10 * np.sin(2 * np.pi * 4 * t)
            
        elif emphasis_pattern == 'sudden':  
            burst = np.exp(-5 * t)
            emphasis = 1.0 + 0.20 * burst
            
        elif emphasis_pattern == 'nasal': 
            emphasis = np.ones(len(audio)) * 0.92
            
        else: 
            emphasis = np.ones(len(audio))
        
        if len(emphasis) > 100:
            window_size = min(int(sr * 0.02), len(emphasis) // 10)
            if window_size > 2:
                emphasis = signal.savgol_filter(emphasis, window_size | 1, 2)
        
        audio = audio * emphasis
        
    except Exception as e:
        print(f"  Emphasis pattern skipped: {e}")
    
    return audio


def apply_spectral_shaping(audio, sr, emotion):
    try:
        if emotion == 'happy':
            b, a = signal.butter(2, [1000, 6000], btype='band', fs=sr)
            boost = signal.filtfilt(b, a, audio) * 0.15
            audio = audio + boost
            
        elif emotion == 'sad':
            b, a = signal.butter(3, 5000, btype='low', fs=sr)
            audio = signal.filtfilt(b, a, audio)
            
        elif emotion == 'angry':
            b, a = signal.butter(2, [800, 3000], btype='band', fs=sr)
            boost = signal.filtfilt(b, a, audio) * 0.20
            audio = audio + boost
            
        elif emotion == 'fearful':
            b, a = signal.butter(2, 3000, btype='high', fs=sr)
            boost = signal.filtfilt(b, a, audio) * 0.12
            audio = audio + boost
            
        elif emotion == 'surprised':
            b, a = signal.butter(2, [500, 7000], btype='band', fs=sr)
            boost = signal.filtfilt(b, a, audio) * 0.18
            audio = audio + boost
            
    except Exception as e:
        print(f"  Spectral shaping skipped: {e}")
    
    return audio


def apply_final_polish(audio, sr):
    audio = audio - np.mean(audio)

    b, a = signal.butter(4, 8000, btype='low', fs=sr)
    audio = signal.filtfilt(b, a, audio)

    threshold = 0.75
    ratio = 2.5
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )
 
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.88
    
    return audio




class EmotionTTSEngine:
    def __init__(self, emotion_model_path='E:/dlpro/emotion_model.keras', tts_model_path='emotion_tts_gan'):
        self.sample_rate = SAMPLE_RATE
        self.tts_model_path = tts_model_path
        self.gan_ready = False
        
        print("=" * 60)
        print("Balanced Emotion TTS Engine")
        print("Distinctive emotions + Natural sound quality")
        print("=" * 60)
        
        print("\n[1/3] Extracting embeddings from emotion_model.keras...")
        self.emotion_embeddings, embed_dim = extract_emotion_embeddings(emotion_model_path)
        
        if self.emotion_embeddings is not None:
            print(f" Using the model's learned emotion representations!")
        else:
            embed_dim = 256
            print(f"Using random embeddings")
        
        print("\n[2/3] Building models...")
        self.conditioner = build_emotion_conditioner(embed_dim)
        self.vocoder = build_vocoder_generator()
        print(f"Conditioner input: mel({MEL_CHANNELS}) + embedding({embed_dim})")
        
        cond_path = 'E:/dlpro/' + tts_model_path + '_conditioner.weights.h5'
        voc_path = 'E:/dlpro/' + tts_model_path + '_vocoder.weights.h5'
        if os.path.exists(cond_path) and os.path.exists(voc_path):
            try:
                self.conditioner.load_weights(cond_path)
                self.vocoder.load_weights(voc_path)
                self.gan_ready = True
                print(f"\n Weights loaded!")
            except Exception as e:
                print(f"\n Could not load weights: {e}")
        
        print("\n[3/3] Loading TTS backend...")
        self.use_edge_tts = self._check_edge_tts()
        if not self.use_edge_tts:
            self._load_pyttsx3()
    
    def _check_edge_tts(self):
        try:
            import edge_tts
            print(" edge-tts backend ready!")
            return True
        except ImportError:
            print("edge-tts not found. Falling back to pyttsx3.")
            return False
    
    def _load_pyttsx3(self):
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self.use_pyttsx3 = True
            print("pyttsx3 fallback ready")
        except:
            self.use_pyttsx3 = False
            print("No TTS backend available")
    
    async def _edge_tts_generate(self, text, emotion, output_path):
        import edge_tts
        profile = EMOTION_VOICE_PROFILES.get(emotion, EMOTION_VOICE_PROFILES['neutral'])
        
        communicate = edge_tts.Communicate(
            text, voice=VOICE,
            rate=profile['rate'],
            pitch=profile['pitch'],
            volume=profile['volume']
        )
        await communicate.save(output_path)
    
    def _apply_gan_conditioning(self, audio, emotion):
        try:
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32), sr=self.sample_rate,
                n_mels=MEL_CHANNELS, hop_length=256, win_length=1024, fmax=8000
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_t = mel_norm[np.newaxis].astype(np.float32)
            
            all_shifts = []
            for idx in range(NUM_EMOTIONS):
                e_t = self.emotion_embeddings[idx][np.newaxis].astype(np.float32)
                cond = self.conditioner([mel_t, e_t], training=False)
                shift = float(np.mean(np.abs(np.array(cond)[0] - mel_norm)))
                all_shifts.append(shift)
            
            emotion_idx = EMOTION_MAP.get(emotion, 0)
            current_shift = all_shifts[emotion_idx]
            max_shift = max(all_shifts) + 1e-8
            min_shift = min(all_shifts)
            shift_range = max_shift - min_shift + 1e-8
            
            normalized = (current_shift - min_shift) / shift_range
            dsp_scale = 0.80 + normalized * 0.4
            
            print(f"Conditioning: scale {dsp_scale:.2f}")
            return audio, dsp_scale
        
        except Exception as e:
            print(f"GAN skipped: {e}")
            return audio, 1.0
    
    def _apply_balanced_emotion_processing(self, audio, emotion, scale_factor=1.0):

        p = EMOTION_VOICE_PROFILES.get(emotion, EMOTION_VOICE_PROFILES['neutral'])
        
        scale = np.clip(scale_factor, 0.7, 1.3)
        
        print(f"Processing: {emotion} (balanced approach)")
        
        pitch_steps = p['pitch_steps'] * scale
        audio = apply_smooth_pitch_shift(audio, self.sample_rate, pitch_steps)
        
        speed = p['speed'] * scale
        audio = apply_timing_variation(audio, self.sample_rate, speed)
        
        variance = p['pitch_variance']
        audio = apply_emotion_prosody(audio, self.sample_rate, emotion, variance)
        
        audio = apply_emphasis_pattern(audio, self.sample_rate, emotion, p['emphasis_pattern'])
        
        audio = apply_spectral_shaping(audio, self.sample_rate, emotion)
        
        audio = audio * p['energy']
    
        audio = apply_final_polish(audio, self.sample_rate)
        
        return audio
    
    def synthesize(self, text, emotion='neutral', output_path='response.wav'):
        print(f"\n{'='*50}")
        print(f"[{emotion.upper()}]")
        print(f"{'='*50}")
        
        try:
            if self.use_edge_tts:
                temp_path = tempfile.mktemp(suffix='.wav')
                
                asyncio.run(self._edge_tts_generate(text, emotion, temp_path))
                audio, sr = librosa.load(temp_path, sr=self.sample_rate)
                
                scale_factor = 1.0
                if self.gan_ready:
                    audio, scale_factor = self._apply_gan_conditioning(audio, emotion)
                
                audio = self._apply_balanced_emotion_processing(audio, emotion, scale_factor)
                
                sf.write(output_path, audio, self.sample_rate)
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            elif self.use_pyttsx3:
                p = EMOTION_VOICE_PROFILES.get(emotion, EMOTION_VOICE_PROFILES['neutral'])
                self.pyttsx3_engine.setProperty('rate', int(180 * p['speed']))
                self.pyttsx3_engine.setProperty('volume', min(1.0, 0.9 * p['energy']))
                self.pyttsx3_engine.save_to_file(text, output_path)
                self.pyttsx3_engine.runAndWait()
                
                audio, sr = librosa.load(output_path, sr=self.sample_rate)
                audio = self._apply_balanced_emotion_processing(audio, emotion)
                sf.write(output_path, audio, self.sample_rate)
            else:
                print("No TTS backend available.")
                return False

            audio_out, sr_out = sf.read(output_path)
            sd.play(audio_out, sr_out)
            sd.wait()
            print(f"Playback complete\n")
            return True
        
        except Exception as e:
            print(f"Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return False

_engine = None

def speak_response(text, emotion='neutral', emotion_model_path='E:/dlpro/emotion_model.keras'):
    global _engine
    if _engine is None:
        _engine = EmotionTTSEngine(emotion_model_path=emotion_model_path)
    _engine.synthesize(text, emotion=emotion)

if __name__ == "__main__":
    import time
    print("=" * 70)
    print("Test sample")
    print("=" * 70)
    
    test_text = "Hello! I am responding to you with emotion."
    
    for emotion in ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgust']:
        speak_response(test_text, emotion=emotion,
                       emotion_model_path='E:/dlpro/emotion_model.keras')
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)