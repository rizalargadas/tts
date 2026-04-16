"""
Text-to-Speech Server using Kokoro TTS
High-quality, free, fully local neural voices — no API key required

Usage:
    pip install kokoro flask flask-cors
    python server.py

Windows: Also install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases
"""

import os
import sys
import struct
import threading
import webbrowser
import io
import re
import numpy as np
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS

if sys.platform == 'win32':
    for path in [r"C:\Program Files\eSpeak NG", r"C:\Program Files (x86)\eSpeak NG"]:
        lib = os.path.join(path, "libespeak-ng.dll")
        if os.path.exists(lib):
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib)
            os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", os.path.join(path, "espeak-ng.exe"))
            break

try:
    from kokoro import KPipeline
except ImportError:
    print("\n  ERROR: Kokoro TTS not installed.")
    print("  Run: pip install kokoro")
    if sys.platform == 'win32':
        print("  Also install espeak-ng: https://github.com/espeak-ng/espeak-ng/releases")
    sys.exit(1)


def get_resource_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


RESOURCE_DIR = get_resource_dir()

app = Flask(__name__)
CORS(app)

VOICES = [
    {"id": "af_heart", "name": "Heart", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "A"},
    {"id": "af_bella", "name": "Bella", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "A-"},
    {"id": "af_nicole", "name": "Nicole", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "B-"},
    {"id": "af_aoede", "name": "Aoede", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C+"},
    {"id": "af_kore", "name": "Kore", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C+"},
    {"id": "af_sarah", "name": "Sarah", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C+"},
    {"id": "af_alloy", "name": "Alloy", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C"},
    {"id": "af_nova", "name": "Nova", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C"},
    {"id": "af_sky", "name": "Sky", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "C-"},
    {"id": "af_jessica", "name": "Jessica", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "D"},
    {"id": "af_river", "name": "River", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Female", "grade": "D"},
    {"id": "am_fenrir", "name": "Fenrir", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "C+"},
    {"id": "am_michael", "name": "Michael", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "C+"},
    {"id": "am_puck", "name": "Puck", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "C+"},
    {"id": "am_echo", "name": "Echo", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "D"},
    {"id": "am_eric", "name": "Eric", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "D"},
    {"id": "am_liam", "name": "Liam", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "D"},
    {"id": "am_onyx", "name": "Onyx", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "D"},
    {"id": "am_adam", "name": "Adam", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "F+"},
    {"id": "am_santa", "name": "Santa", "lang": "a", "lang_name": "American English", "locale": "en-US", "gender": "Male", "grade": "D-"},
    {"id": "bf_emma", "name": "Emma", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Female", "grade": "B-"},
    {"id": "bf_isabella", "name": "Isabella", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Female", "grade": "C"},
    {"id": "bf_alice", "name": "Alice", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Female", "grade": "D"},
    {"id": "bf_lily", "name": "Lily", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Female", "grade": "D"},
    {"id": "bm_george", "name": "George", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Male", "grade": "C"},
    {"id": "bm_fable", "name": "Fable", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Male", "grade": "C"},
    {"id": "bm_lewis", "name": "Lewis", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Male", "grade": "D+"},
    {"id": "bm_daniel", "name": "Daniel", "lang": "b", "lang_name": "British English", "locale": "en-GB", "gender": "Male", "grade": "D"},
    {"id": "jf_alpha", "name": "Alpha", "lang": "j", "lang_name": "Japanese", "locale": "ja-JP", "gender": "Female", "grade": "C+"},
    {"id": "jf_gongitsune", "name": "Gongitsune", "lang": "j", "lang_name": "Japanese", "locale": "ja-JP", "gender": "Female", "grade": "C"},
    {"id": "jf_nezumi", "name": "Nezumi", "lang": "j", "lang_name": "Japanese", "locale": "ja-JP", "gender": "Female", "grade": "C-"},
    {"id": "jf_tebukuro", "name": "Tebukuro", "lang": "j", "lang_name": "Japanese", "locale": "ja-JP", "gender": "Female", "grade": "C"},
    {"id": "jm_kumo", "name": "Kumo", "lang": "j", "lang_name": "Japanese", "locale": "ja-JP", "gender": "Male", "grade": "C-"},
    {"id": "zf_xiaobei", "name": "Xiaobei", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Female", "grade": "D"},
    {"id": "zf_xiaoni", "name": "Xiaoni", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Female", "grade": "D"},
    {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Female", "grade": "D"},
    {"id": "zf_xiaoyi", "name": "Xiaoyi", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Female", "grade": "D"},
    {"id": "zm_yunjian", "name": "Yunjian", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Male", "grade": "D"},
    {"id": "zm_yunxi", "name": "Yunxi", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Male", "grade": "D"},
    {"id": "zm_yunxia", "name": "Yunxia", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Male", "grade": "D"},
    {"id": "zm_yunyang", "name": "Yunyang", "lang": "z", "lang_name": "Chinese", "locale": "zh-CN", "gender": "Male", "grade": "D"},
    {"id": "ef_dora", "name": "Dora", "lang": "e", "lang_name": "Spanish", "locale": "es-ES", "gender": "Female", "grade": "C"},
    {"id": "em_alex", "name": "Alex", "lang": "e", "lang_name": "Spanish", "locale": "es-ES", "gender": "Male", "grade": "C"},
    {"id": "em_santa", "name": "Santa", "lang": "e", "lang_name": "Spanish", "locale": "es-ES", "gender": "Male", "grade": "D-"},
    {"id": "ff_siwis", "name": "Siwis", "lang": "f", "lang_name": "French", "locale": "fr-FR", "gender": "Female", "grade": "B-"},
    {"id": "hf_alpha", "name": "Alpha", "lang": "h", "lang_name": "Hindi", "locale": "hi-IN", "gender": "Female", "grade": "C"},
    {"id": "hf_beta", "name": "Beta", "lang": "h", "lang_name": "Hindi", "locale": "hi-IN", "gender": "Female", "grade": "C"},
    {"id": "hm_omega", "name": "Omega", "lang": "h", "lang_name": "Hindi", "locale": "hi-IN", "gender": "Male", "grade": "C"},
    {"id": "hm_psi", "name": "Psi", "lang": "h", "lang_name": "Hindi", "locale": "hi-IN", "gender": "Male", "grade": "C"},
    {"id": "if_sara", "name": "Sara", "lang": "i", "lang_name": "Italian", "locale": "it-IT", "gender": "Female", "grade": "C"},
    {"id": "im_nicola", "name": "Nicola", "lang": "i", "lang_name": "Italian", "locale": "it-IT", "gender": "Male", "grade": "C"},
    {"id": "pf_dora", "name": "Dora", "lang": "p", "lang_name": "Portuguese", "locale": "pt-BR", "gender": "Female", "grade": "C"},
    {"id": "pm_alex", "name": "Alex", "lang": "p", "lang_name": "Portuguese", "locale": "pt-BR", "gender": "Male", "grade": "C"},
    {"id": "pm_santa", "name": "Santa", "lang": "p", "lang_name": "Portuguese", "locale": "pt-BR", "gender": "Male", "grade": "D-"},
]

pipelines = {}


def get_pipeline(lang_code):
    if lang_code not in pipelines:
        print(f"  Loading pipeline for language: {lang_code}...")
        pipelines[lang_code] = KPipeline(lang_code=lang_code)
        print(f"  Pipeline loaded: {lang_code}")
    return pipelines[lang_code]


CHUNK_THRESHOLD = 5000


def split_text_into_chunks(text, max_size=CHUNK_THRESHOLD):
    if len(text) <= max_size:
        return [text]

    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence if current else sentence

    if current:
        chunks.append(current.strip())

    final = []
    for chunk in chunks:
        if len(chunk) > max_size:
            words = chunk.split()
            sub = ""
            for word in words:
                if len(sub) + len(word) + 1 > max_size and sub:
                    final.append(sub.strip())
                    sub = word
                else:
                    sub += " " + word if sub else word
            if sub:
                final.append(sub.strip())
        else:
            final.append(chunk)

    return final


def audio_to_wav_bytes(audio_np, sample_rate=24000):
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    data = audio_int16.tobytes()

    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + len(data)))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', len(data)))
    buf.write(data)
    buf.seek(0)
    return buf


def generate_audio(text, voice_id, speed=1.0):
    voice_info = next((v for v in VOICES if v["id"] == voice_id), None)
    if not voice_info:
        raise ValueError(f"Unknown voice: {voice_id}")

    lang_code = voice_info["lang"]
    pipeline = get_pipeline(lang_code)

    chunks = split_text_into_chunks(text)
    all_audio = []

    for chunk in chunks:
        for _gs, _ps, audio in pipeline(chunk, voice=voice_id, speed=speed):
            if audio is not None:
                all_audio.append(audio)

    if not all_audio:
        raise ValueError("No audio generated")

    combined = np.concatenate(all_audio)
    return audio_to_wav_bytes(combined)


@app.route('/')
def index():
    return send_from_directory(RESOURCE_DIR, 'index.html')


@app.route('/api/voices')
def list_voices():
    return jsonify(VOICES)


@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'af_heart')
    speed = float(data.get('speed', 1.0))

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) > 50000:
        return jsonify({'error': 'Text too long (max 50000 characters)'}), 400

    try:
        wav_buffer = generate_audio(text, voice, speed)
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='speech.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/download', methods=['POST'])
def download_speech():
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'af_heart')
    speed = float(data.get('speed', 1.0))

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        wav_buffer = generate_audio(text, voice, speed)
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='speech.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Kokoro TTS Server")
    print("  High-quality local neural voices")
    print("=" * 50)
    print("\n  Loading model (first run downloads ~300MB)...")

    get_pipeline('a')

    print("  Model ready!")
    print(f"\n  Open http://localhost:5000 in your browser\n")
    print("  Press Ctrl+C to stop the server\n")

    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=not is_frozen, threaded=True)
