#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Ollama Model Arena: .venv was not found."
  echo "Run installation steps in README.md first."
  echo
  read -r -p "Press Enter to close..."
  exit 1
fi

export OMA_SERVER_NAME="0.0.0.0"
export OMA_SERVER_PORT="7860"

echo "Starting Ollama Model Arena — Remote / Tailscale mode"
echo "Local: http://127.0.0.1:${OMA_SERVER_PORT}"
echo "Remote: http://<Mac-Tailscale-IP>:${OMA_SERVER_PORT}"
echo "Keep this Terminal window open while using the Arena."
echo
if command -v caffeinate >/dev/null 2>&1; then
  echo "Sleep prevention: caffeinate enabled"
  exec caffeinate -i .venv/bin/python app.py
else
  exec .venv/bin/python app.py
fi
