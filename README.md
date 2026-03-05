# Whisper Dictation (ndw)

High-accuracy Czech dictation using OpenAI Whisper `large-v3-turbo` model accelerated by NVIDIA GPU.

**Forked from:** [ideasman42/nerd-dictation](https://github.com/ideasman42/nerd-dictation) - modified for GPU-accelerated Czech transcription using faster-whisper.

## Requirements

### System packages
- **Audio**: pipewire (or pulseaudio), pw-cat
- **Input**: xdotool (for typing)
- **Sounds**: sox (for status beeps)

Install on Arch:
```bash
sudo pacman -S pipewire xdotool sox
```

Install on Debian/Ubuntu:
```bash
sudo apt install pipewire xdotool sox
```

### Python
- **uv** (package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Installation

```bash
cd /home/michael/work/src/whisper-dictation
uv sync
```

Symlinks are already in `~/bin`:
- `ndw` - Start dictation
- `ndw-pause` - Toggle pause/resume (SIGUSR1)
- `ndw-status` - Show current status

## Usage

### Basic
```bash
ndw                    # Start dictation
ndw --device cpu       # Force CPU (no GPU)
ndw --offline          # Offline mode (no network)
```

### Models
```bash
ndw                                    # Default: large-v3-turbo (fast)
ndw --model Systran/faster-whisper-large-v3  # Most accurate
```

### Diagnostics
```bash
ndw --doctor        # Check system and model status
ndw --clear-cache   # Delete cached models
ndw --reload        # Force model download
```

### Controls
- **Ctrl+C** - Exit
- **p + Enter** - Resume if paused
- **Global hotkey** (via ndw-pause) - Toggle pause/resume

## Status Sounds
- **High pitch (0.1s)**: Resume / Start (Ready to speak)
- **Low pitch (0.4s)**: Pause (Recording stopped)

## Using just (optional tasks)

```bash
just          # List all commands
just sync     # uv sync
just doctor   # System check
just cache    # Show cached models
just clean    # Clear cache
just turbo    # Run with turbo model
just cpu      # Run with CPU
```

## Recent Improvements
- **Model Upgrade**: Using `large-v3-turbo` by default for faster response times
- **Pre-roll Buffer**: 300ms buffer prevents clipping first words after pause
- **Custom VAD**: Disabled internal Whisper VAD to avoid aggressive speech cutting
- **Self-checking**: ndw.sh auto-validates environment and runs uv sync if needed
