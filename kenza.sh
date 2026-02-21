#!/bin/bash
# Kenza launcher - suppresses JACK error messages
# Run with: ./kenza.sh [options]
#
# Options:
#   --no-wake   Skip wake word, always listen
#   --text      Text mode only
#   --test-mic  Test microphone

python kenza_conversation.py "$@" 2>/dev/null
