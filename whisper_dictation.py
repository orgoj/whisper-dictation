#!/usr/bin/env python3
import argparse
import os
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

# FILTERED PHRASES - text in this list will not be typed
FILTERED_PHRASES = [
    "Děkujeme za pozornost.",
    "Děkujeme.",
    "Děkuji za pozornost.",
    "Konec.",
    "Titulky vytvořil Jirka Kováč",
    "Titulky vytvořil JohnyX.",
    "www.arkance-systems.cz",
    "www.hradeckralove.org",
]


# Quick startup check for critical tools
def quick_startup_check():
    """Check critical tools before starting. Warn but don't exit."""
    missing = []
    for tool in ["pw-cat", "xdotool", "play"]:
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            missing.append(tool)
    if missing:
        sys.stderr.write(f"[WARNING] Missing tools: {', '.join(missing)}\n")
        sys.stderr.write("  Install: sudo pacman -S pipewire xdotool sox\n")


# 1. PÍPNUTÍ
def play_pip(action):
    try:
        freq = 800 if action == "resume" else 400
        duration = 0.1 if action == "resume" else 0.4
        os.system(f"play -q -n synth {duration} sine {freq} > /dev/null 2>&1 &")
    except:
        pass


class AudioReader(threading.Thread):
    """Continuously reads audio from pipe so no data is lost during transcription."""

    def __init__(self, proc_stdout, sample_rate, chunk_duration=0.1):
        super().__init__(daemon=True)
        self.proc_stdout = proc_stdout
        self.chunk_bytes = int(
            sample_rate * chunk_duration * 2
        )  # 16-bit = 2 bytes/sample
        self._lock = threading.Lock()
        self._chunks = []
        self._last_rms = 0.0
        self._running = True
        self._eof = False

    def run(self):
        while self._running:
            try:
                raw = self.proc_stdout.read(self.chunk_bytes)
            except ValueError:
                break
            if not raw:
                self._eof = True
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            with self._lock:
                self._chunks.append(chunk)
                self._last_rms = rms

    def take_audio(self):
        """Return all accumulated audio and clear internal buffer."""
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            result = np.concatenate(self._chunks)
            self._chunks.clear()
            return result

    @property
    def last_rms(self):
        with self._lock:
            return self._last_rms

    @property
    def eof(self):
        return self._eof

    def stop(self):
        self._running = False


def type_text(text, args):
    if not text:
        return
    stripped_text = text.strip()
    if stripped_text in FILTERED_PHRASES:
        sys.stderr.write(f" [FILTERED EXACT PHRASE: {stripped_text}]\n")
        return
    sys.stderr.write(f" [TYPE] {text}\n")
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text], check=False
    )


def run_dictation(args):
    quick_startup_check()

    # State management using local variables
    class State:
        paused = False
        exit_requested = False

    state = State()

    # Signal-safe write function
    def safe_write(msg):
        """Write to stderr using os.write (signal-safe)"""
        try:
            os.write(2, f"{msg}\n".encode())
        except:
            pass

    def update_status_file():
        status = "PAUSED" if state.paused else "LIVE"
        try:
            with open("/tmp/ndw.status", "w") as f:
                f.write(status)
            os.chmod("/tmp/ndw.status", 0o666)
        except Exception as e:
            safe_write(f"Error updating status file: {e}")

    def toggle_pause(signum=None, frame=None):
        state.paused = not state.paused
        status = "PAUSED" if state.paused else "RESUMED"
        timestamp = time.strftime("%H:%M:%S")
        safe_write(f"\n--- {status} {timestamp} ---")
        play_pip("pause" if state.paused else "resume")
        update_status_file()

    def handle_sigint(signum, frame):
        safe_write("\n--- EXITING... ---")
        state.exit_requested = True

    # SIGINT (Ctrl+C) is critical.
    # Python's default behavior is to raise KeyboardInterrupt.
    # By setting a handler, we override this, but we MUST make sure
    # the main loop doesn't collapse.
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGUSR1, toggle_pause)

    sys.stderr.write("--- ACTIVATING AUDIO ---\n")
    play_pip("resume")
    update_status_file()
    cmd = [
        "pw-cat",
        "--record",
        "--format=s16",
        "--rate",
        str(args.sample_rate),
        "--channels=1",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr)

    sys.stderr.write(f"--- LOADING MODEL ({args.model.upper()}) ---\n")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        local_files_only=args.offline,
    )
    sys.stderr.write("--- SYSTEM READY (ndw-pause to toggle) ---\n")
    sys.stderr.write(
        "--- Controls: Ctrl+C = EXIT | 'p' ENTER = RESUME (if paused via ndw-pause) ---\n"
    )

    # Audio reader thread - reads from pipe continuously, even during transcription
    reader = AudioReader(proc.stdout, args.sample_rate)
    reader.start()

    audio_buffer = np.array([], dtype=np.float32)
    # Rolling buffer of the last 500ms of silence
    silence_buffer = np.array([], dtype=np.float32)
    silence_limit_samples = int(args.sample_rate * 0.5)

    recording_started = False
    last_process_time = time.time()

    # THE MAIN LOOP must be extremely resilient
    while not state.exit_requested:
        try:
            if reader.eof:
                break

            # select ONLY on stdin for keyboard commands, 0.1s timeout
            r, _, _ = select.select([sys.stdin], [], [], 0.1)

            if sys.stdin in r:
                line = sys.stdin.readline().lower()
                if "p" in line and state.paused:
                    state.paused = False
                    sys.stderr.write("\n--- RESUMED ---\n")
                    play_pip("resume")
                    update_status_file()
                elif "q" in line:
                    state.exit_requested = True

            now = time.time()

            if state.paused:
                reader.take_audio()  # discard audio during pause
                audio_buffer = np.array([], dtype=np.float32)
                silence_buffer = np.array([], dtype=np.float32)
                recording_started = False
                last_process_time = now
                continue

            # Grab ALL audio accumulated since last check
            new_audio = reader.take_audio()
            rms = 0.0

            if len(new_audio) > 0:
                rms = float(np.sqrt(np.mean(new_audio**2)))

                if rms < args.vad_threshold and not recording_started:
                    # Keep updating silence buffer (rolling window)
                    silence_buffer = np.append(silence_buffer, new_audio)
                    if len(silence_buffer) > silence_limit_samples:
                        silence_buffer = silence_buffer[-silence_limit_samples:]
                else:
                    # We are recording or just started
                    if not recording_started:
                        # PREPEND SILENCE BUFFER ONCE
                        audio_buffer = np.append(silence_buffer, new_audio)
                        recording_started = True
                        silence_buffer = np.array([], dtype=np.float32)
                    else:
                        audio_buffer = np.append(audio_buffer, new_audio)

                if rms >= args.vad_threshold:
                    last_process_time = now

            if (
                recording_started
                and (now - last_process_time > args.silence_limit)
            ) or len(audio_buffer) > args.sample_rate * args.buffer_limit:

                if len(audio_buffer) > args.sample_rate * 0.5:
                    segments, _ = model.transcribe(
                        audio_buffer,
                        language=args.lang,
                        beam_size=args.beam_size,
                        vad_filter=True,
                        initial_prompt=args.prompt,
                    )
                    phrase = " ".join(
                        [s.text for s in segments if s.no_speech_prob < 0.4]
                    ).strip()
                    if phrase:
                        type_text(phrase + " ", args)

                audio_buffer = np.array([], dtype=np.float32)
                recording_started = False
                last_process_time = time.time()

            if args.debug or (int(now * 5) % 5 == 0):
                sys.stderr.write(
                    f"\r[LIVE] Vol: {rms:.4f} | Buf: {len(audio_buffer) / args.sample_rate:.1f}s "
                )
                sys.stderr.flush()

        except (KeyboardInterrupt, InterruptedError):
            # The signal handler has already run and updated state.
            # We just need to keep the loop going.
            continue
        except Exception as e:
            if not state.exit_requested:
                sys.stderr.write(f"\nCRITICAL ERROR: {e}\n")
                break

    reader.stop()
    proc.terminate()
    try:
        if os.path.exists("/tmp/ndw.status"):
            os.remove("/tmp/ndw.status")
    except:
        pass
    sys.stderr.write("\n--- DISCONNECTED ---\n")


def check_system(args):
    """Diagnostika systému a modelů."""
    import os
    from datetime import datetime
    from huggingface_hub import scan_cache_dir, model_info

    sys.stderr.write("--- NDW DOCTOR ---\n")

    # 1. Kontrola nástrojů
    for tool in ["pw-cat", "xdotool", "play"]:
        res = subprocess.run(["which", tool], capture_output=True, text=True)
        if res.returncode == 0:
            sys.stderr.write(f"[OK] Tool found: {tool} ({res.stdout.strip()})\n")
        else:
            sys.stderr.write(f"[!!] Tool MISSING: {tool}\n")

    # 2. Výpis všech modelů v cache
    sys.stderr.write("\nScanning HuggingFace cache for faster-whisper models...\n")
    try:
        cache_info = scan_cache_dir()
        fw_repos = [r for r in cache_info.repos if "whisper" in r.repo_id.lower()]

        if not fw_repos:
            sys.stderr.write("No Whisper models found in cache.\n")
        else:
            for repo in fw_repos:
                size_gb = repo.size_on_disk / (1024**3)
                last_mod = datetime.fromtimestamp(repo.last_modified).strftime('%Y-%m-%d %H:%M:%S')
                is_current = " [CURRENT]" if args.model in repo.repo_id else ""
                sys.stderr.write(f"- {repo.repo_id}{is_current}\n")
                sys.stderr.write(f"  Size: {size_gb:.2f} GB | Modified: {last_mod}\n")

                # Pokud je to aktuální model, zkusíme remote update check
                if is_current and not args.offline:
                    try:
                        local_hash = list(repo.revisions)[0].commit_hash
                        remote_info = model_info(repo.repo_id)
                        if remote_info.sha != local_hash:
                            sys.stderr.write(f"  [!!] NEW VERSION AVAILABLE! (Remote: {remote_info.sha[:8]})\n")
                        else:
                            sys.stderr.write("  [OK] Model is up to date.\n")
                    except:
                        pass

        if args.model not in [r.repo_id for r in fw_repos] and not any(args.model in r.repo_id for r in fw_repos):
            sys.stderr.write(f"\n[!!] Target model '{args.model}' NOT found in cache.\n")
            sys.stderr.write("     Run with --reload to download it.\n")

    except Exception as e:
        sys.stderr.write(f"[??] Could not scan HuggingFace cache: {e}\n")

    sys.stderr.write("\n--- DOCTOR FINISHED ---\n")
    sys.exit(0)


def clear_cache():
    """Interaktivní promazání cache modelů."""
    from huggingface_hub import scan_cache_dir
    try:
        cache_info = scan_cache_dir()
        fw_repos = [r for r in cache_info.repos if "whisper" in r.repo_id.lower()]

        if not fw_repos:
            print("No Whisper models found to delete.")
            return

        print("\n--- CACHE CLEANUP ---")
        for i, repo in enumerate(fw_repos):
            size_gb = repo.size_on_disk / (1024**3)
            print(f"[{i}] {repo.repo_id} ({size_gb:.2f} GB)")

        print("[a] DELETE ALL")
        print("[q] CANCEL")

        choice = input("\nSelect model to DELETE (number/a/q): ").strip().lower()

        if choice == 'q':
            return

        delete_strategy = None
        if choice == 'a':
            delete_strategy = cache_info.delete_revisions(*[rev.commit_hash for repo in fw_repos for rev in repo.revisions])
        elif choice.isdigit() and int(choice) < len(fw_repos):
            repo = fw_repos[int(choice)]
            delete_strategy = cache_info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])

        if delete_strategy:
            print(f"Deleting... Freed {delete_strategy.expected_freed_size_str}")
            delete_strategy.execute()
        else:
            print("Invalid choice.")

    except Exception as e:
        print(f"Error clearing cache: {e}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Dictation CLI")
    parser.add_argument(
        "--model",
        default="Systran/faster-whisper-large-v3",
        help="Model (tiny, base, small, medium, large-v3, large-v3-turbo). "
             "Doporučený pro přesnost je 'large-v3', pro rychlost 'large-v3-turbo'."
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for transcription"
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--compute-type", default="int8", help="int8, float16, etc.")
    parser.add_argument("--lang", default="cs", help="Language code")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Audio sample rate"
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=0.01, help="RMS threshold for silence"
    )
    parser.add_argument(
        "--silence-limit",
        type=float,
        default=0.8,
        help="Seconds of silence to trigger transcription",
    )
    parser.add_argument(
        "--buffer-limit",
        type=float,
        default=7.0,
        help="Max seconds of audio before forced transcription",
    )
    parser.add_argument("--offline", action="store_true", help="Force offline mode")
    parser.add_argument("--reload", action="store_true", help="Force model download/update")
    parser.add_argument(
        "--doctor", action="store_true", help="Check system and model status"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Interactively delete models from cache"
    )
    parser.add_argument(
        "--prompt", default="Programátorské diktování v češtině. Termíny jako default, AGENTS.md, claude, code, class, function, git, repository zůstávají v angličtině.", help="Initial prompt for Whisper"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Zobrazit podrobné ladicí informace"
    )
    args = parser.parse_args()

    # Pokud je reload, vypneme offline režim, aby se to mohlo stáhnout
    if args.reload:
        args.offline = False

    if args.clear_cache:
        clear_cache()

    if args.doctor:
        check_system(args)

    run_dictation(args)
