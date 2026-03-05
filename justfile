# whisper-dictation tasks

default:
    @just --list

# Install/update dependencies
sync:
    uv sync

# Run dictation (foreground)
run:
    ./ndw.sh

# Check system and model status
doctor:
    ./ndw.sh --doctor

# Show cached Whisper models
cache:
    @uv run python -c "from huggingface_hub import scan_cache_dir; from datetime import datetime; info = scan_cache_dir(); whisper = [r for r in info.repos if 'whisper' in r.repo_id.lower()]; _ = [print(f'{r.repo_id}: {r.size_on_disk/1024**3:.2f}GB') for r in whisper] if whisper else print('No models found')"

# Clear model cache interactively
clean:
    ./ndw.sh --clear-cache

# Force model re-download
reload:
    ./ndw.sh --reload

# Run with CPU instead of GPU
cpu:
    ./ndw.sh --device cpu

# Run with large-v3-turbo (faster)
turbo:
    ./ndw.sh --model Systran/faster-whisper-large-v3-turbo

# Run with large-v3 (most accurate)
accurate:
    ./ndw.sh --model Systran/faster-whisper-large-v3

# Install system dependencies (Arch/Debian)
install-deps:
    #!/bin/bash
    if command -v pacman &> /dev/null; then
        echo "Installing Arch Linux dependencies..."
        sudo pacman -S --needed python-pipewire wireplumber pipewire-pulse xdotool sox
    elif command -v apt &> /dev/null; then
        echo "Installing Debian/Ubuntu dependencies..."
        sudo apt install -y pipewire xdotool sox libsox-dev
    else
        echo "Unknown distro. Install: pipewire/pulseaudio, xdotool, sox"
    fi
