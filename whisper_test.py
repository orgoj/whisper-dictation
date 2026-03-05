
import sys
import time
from faster_whisper import WhisperModel

model_size = "small"

print(f"Loading model '{model_size}'...", file=sys.stderr)
start = time.time()
# Run on GPU
model = WhisperModel(model_size, device="cuda", compute_type="int8")
print(f"Model loaded in {time.time() - start:.2f}s", file=sys.stderr)

def transcribe(audio_path):
    start = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5, language="cs")
    
    text = ""
    for segment in segments:
        text += segment.text
    
    print(f"Transcribed in {time.time() - start:.2f}s", file=sys.stderr)
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(transcribe(sys.argv[1]))
