
import os
import sys
import socket
import subprocess
import time
from faster_whisper import WhisperModel

# Configuration
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
SOCKET_PATH = "/tmp/whisper_dictation.sock"

def type_text(text):
    if not text:
        return
    subprocess.run(["xdotool", "type", "--clearmodifiers", "--", text])

def main():
    print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...", file=sys.stderr)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded and ready.", file=sys.stderr)

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(1024).decode('utf-8').strip()
                if not data:
                    continue
                
                if data == "QUIT":
                    break
                
                audio_path = data
                if os.path.exists(audio_path):
                    # Transcribe
                    segments, info = model.transcribe(audio_path, beam_size=5, language="cs")
                    text = " ".join([s.text for s in segments]).strip()
                    
                    if text:
                        print(f"Result: {text}", file=sys.stderr)
                        type_text(text)
                    else:
                        print("No speech detected.", file=sys.stderr)
                
                conn.sendall(b"OK")
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
            finally:
                conn.close()
    finally:
        os.remove(SOCKET_PATH)

if __name__ == "__main__":
    main()
