#!/bin/bash
# Render build script — installs ffmpeg + Python dependencies
set -e

echo "Installing ffmpeg..."
apt-get update -qq && apt-get install -y ffmpeg -qq

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build complete ✓"
