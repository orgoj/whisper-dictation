#!/bin/bash
set -e
WHISPER_DIR="/home/michael/work/src/whisper-dictation"
cd "$WHISPER_DIR" || exit 1

# Self-check: verify uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' command not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Self-check: verify .venv exists, create if needed
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running 'uv sync'..." >&2
    uv sync
fi

# Self-check: verify dependencies are installed
if ! uv run python -c "import faster_whisper" 2>/dev/null; then
    echo "Dependencies missing or broken. Running 'uv sync'..." >&2
    uv sync
fi

# Run the application
exec uv run --no-sync python whisper_dictation.py "$@"
