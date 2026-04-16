"""
Text-to-Speech Server using Microsoft Edge TTS
Free, unlimited, high-quality neural voices - no API key required

Usage:
    pip install edge-tts flask flask-cors
    python server.py

Then open http://localhost:5000 in your browser
"""

import asyncio
import re
import sys
import threading
import webbrowser
import edge_tts
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
import io
import os

def get_resource_dir():
    """Get the directory containing bundled resources (for PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

RESOURCE_DIR = get_resource_dir()

app = Flask(__name__)
CORS(app)

# Cache voices list
VOICES_CACHE = None

async def get_voices():
    """Get all available voices"""
    global VOICES_CACHE
    if VOICES_CACHE is None:
        VOICES_CACHE = await edge_tts.list_voices()
    return VOICES_CACHE

CHUNK_THRESHOLD = 5000

def split_text_into_chunks(text, max_size=CHUNK_THRESHOLD):
    """Split text at sentence boundaries for parallel processing"""
    if len(text) <= max_size:
        return [text]

    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle sentences longer than max_size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size:
            words = chunk.split()
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) + 1 > max_size and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = word
                else:
                    sub_chunk += " " + word if sub_chunk else word
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks

async def generate_audio(text, voice, rate="+0%", pitch="+0Hz"):
    """Generate audio using edge-tts"""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

async def generate_audio_chunked(text, voice, rate="+0%", pitch="+0Hz"):
    """Generate audio with automatic chunking for large texts"""
    chunks = split_text_into_chunks(text)
    if len(chunks) == 1:
        return await generate_audio(text, voice, rate, pitch)

    tasks = [generate_audio(chunk, voice, rate, pitch) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return b"".join(results)

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory(RESOURCE_DIR, 'index.html')

@app.route('/api/voices')
def list_voices():
    """List all available voices"""
    voices = asyncio.run(get_voices())
    return jsonify(voices)

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'en-US-JennyNeural')
    rate = data.get('rate', '+0%')
    pitch = data.get('pitch', '+0Hz')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if len(text) > 50000:
        return jsonify({'error': 'Text too long (max 50000 characters)'}), 400

    try:
        audio_data = asyncio.run(generate_audio_chunked(text, voice, rate, pitch))
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='speech.mp3'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts/download', methods=['POST'])
def download_speech():
    """Convert text to speech and download"""
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'en-US-JennyNeural')
    rate = data.get('rate', '+0%')
    pitch = data.get('pitch', '+0Hz')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        audio_data = asyncio.run(generate_audio_chunked(text, voice, rate, pitch))
        return send_file(
            io.BytesIO(audio_data),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='speech.mp3'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    """Open browser after a short delay to let the server start."""
    import time
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Microsoft Edge TTS Server")
    print("  Free, unlimited, high-quality neural voices")
    print("="*50)
    print("\n  Open http://localhost:5000 in your browser\n")
    print("  Press Ctrl+C to stop the server\n")

    is_frozen = getattr(sys, 'frozen', False)

    # Auto-open browser when running as exe
    if is_frozen:
        threading.Thread(target=open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=not is_frozen, threaded=True)
