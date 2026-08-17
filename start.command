#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Ollama Model Arena: .venv was not found."
  echo "Run these commands first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -r requirements.txt"
  echo
  read -r -p "Press Enter to close..."
  exit 1
fi

exec .venv/bin/python app.py
