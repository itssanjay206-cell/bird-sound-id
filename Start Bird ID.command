#!/bin/zsh
# Double-click this file to start the Bird Sound ID website.
cd "$(dirname "$0")"
[ -d .venv ] || sh setup.sh
python3 web_app.py
