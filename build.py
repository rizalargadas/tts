"""
Build script to create a standalone TTS application exe.

Usage:
    pip install pyinstaller
    python build.py

Output: dist/TextToSpeech.exe
"""

import subprocess
import sys

def build():
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--name', 'TextToSpeech',
        '--add-data', 'index.html;.',
        '--console',       # keep console so user can see server logs & Ctrl+C
        '--icon', 'NONE',
        '--noconfirm',
        'server.py'
    ]

    print("Building TextToSpeech.exe ...")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 50)
    print("  Build complete!")
    print("  Your exe is at: dist/TextToSpeech.exe")
    print("=" * 50)
    print("\nDouble-click TextToSpeech.exe to launch.")
    print("It will start the server and open your browser automatically.")

if __name__ == '__main__':
    build()
