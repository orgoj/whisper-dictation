#!/bin/bash
STATUS_FILE="/tmp/ndw.status"

# Kontrola, zda proces vůbec běží
if ! pgrep -f whisper_dictation.py > /dev/null; then
    echo ""
    exit 0
fi

if [ ! -f "$STATUS_FILE" ]; then
    echo "❓" # Běží, ale stav neznámý
    exit 0
fi

STATUS=$(cat "$STATUS_FILE")

if [ "$STATUS" = "LIVE" ]; then
    echo -n "n"
elif [ "$STATUS" = "PAUSED" ]; then
    echo -n "g"
else
    echo -n ""
fi
