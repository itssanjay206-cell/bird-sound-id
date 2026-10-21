#!/bin/sh
# One-time setup for Mac/Linux: creates .venv and installs everything (~1-2 GB, a few minutes).
cd "$(dirname "$0")"
python3 -m venv .venv || { echo "Python 3 is needed: https://www.python.org/downloads/"; exit 1; }
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo ""
echo "Setup finished. Start the website with:  python3 web_app.py   (or double-click 'Start Bird ID.command')"
