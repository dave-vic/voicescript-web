#!/bin/bash
# VoiceScript Web — quick start script
# Run this once to launch the app, then open http://localhost:8000

echo ""
echo "  Starting VoiceScript Web…"
echo ""

# Try to find Python 3
PYTHON=""
for p in python3 /opt/anaconda3/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if command -v "$p" &>/dev/null; then
    PYTHON="$p"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "  ✗ Python 3 not found. Install it from https://python.org"
  exit 1
fi

echo "  Using $PYTHON"
echo ""

# Install dependencies if needed
$PYTHON -c "import fastapi" 2>/dev/null || {
  echo "  Installing dependencies…"
  $PYTHON -m pip install -r requirements.txt -q
}

# Launch
$PYTHON main.py
