#!/bin/sh
# Upload the app to a Hugging Face Space (needs `hf auth login` once).
# Usage: sh deploy_hf.sh <hf-username>
set -e
cd "$(dirname "$0")"
USER_NAME="$1"
[ -n "$USER_NAME" ] || { echo "Usage: sh deploy_hf.sh <hf-username>"; exit 1; }
STAGE="$(mktemp -d)"
cp -R bird_sound_analyzer.py web_app.py web "$STAGE"/
cp hf_space/Dockerfile hf_space/requirements-server.txt hf_space/README.md "$STAGE"/
HF=".venv/bin/hf"; [ -x "$HF" ] || HF="hf"
"$HF" repos create "$USER_NAME/bird-sound-id" --type space --space-sdk docker --public --exist-ok
"$HF" upload "$USER_NAME/bird-sound-id" "$STAGE" . --repo-type space --commit-message "Deploy Bird Sound ID"
rm -rf "$STAGE"
echo ""
echo "Deployed. App link: https://$USER_NAME-bird-sound-id.hf.space"
