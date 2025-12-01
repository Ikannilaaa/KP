import os
import wave
from google import genai
from google.genai import types
from deep_translator import GoogleTranslator

API_KEY = "AIzaSyDgWMbV7sZzIA8X3vc3S_zisX2t8vNh1mM" 
client = genai.Client(api_key=API_KEY)

AUDIO_OUTPUT_DIR = os.path.join(os.getcwd(), 'output_audio')
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

def save_wave_file(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """Menyimpan data PCM raw ke file WAV."""
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return filename

def process_tts(text, target_lang='id'):
    try:
        # 1. Translate
        if not text:
            return {"error": "Text kosong"}
            
        print(f"Menerjemahkan ke {target_lang}...")
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        
        # 2. Generate Audio via Gemini
        print("Generate Audio via Gemini...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts", # Pastikan model ini tersedia di akunmu
            contents=translated_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                )
            )
        )
        
        # Ambil data binary
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        # Simpan ke file
        filename = f"tts_{target_lang}_{os.urandom(4).hex()}.wav"
        save_wave_file(filename, audio_data)
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "audio_filename": filename
        }
        
    except Exception as e:
        print(f"Error TTS: {e}")
        return {"error": str(e)}

def get_supported_languages():
    # Mengambil list bahasa dari deep-translator
    lang_map = GoogleTranslator(source="auto", target="en").get_supported_languages(as_dict=True)
    # Format untuk dropdown frontend: [Label, Value]
    return sorted([{"name": name.title(), "code": code} for name, code in lang_map.items()], key=lambda x: x['name'])