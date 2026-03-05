#!/bin/bash
# Najde PID procesu whisper_dictation.py a pošle mu SIGUSR1
PID=$(pgrep -f whisper_dictation.py)

if [ -n "$PID" ]; then
    kill -USR1 $PID
    echo "ndw pause/resume signal sent to PID $PID"
else
    echo "ndw is not running."
fi
