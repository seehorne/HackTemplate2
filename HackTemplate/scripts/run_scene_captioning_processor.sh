#!/bin/bash
echo "Attempting to start scene_captioning_processor..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate whatsai; then echo "Failed to activate conda: whatsai" >&2; exit 1; fi
echo "Conda env 'whatsai' activated for scene_captioning_processor."
cd "/home/runner/work/HackTemplate2/HackTemplate2/HackTemplate"
echo "Starting uvicorn for processors.scene_captioning_processor:app on 127.0.0.1:8003..."
mkdir -p "/home/runner/work/HackTemplate2/HackTemplate2/HackTemplate/logs"
exec uvicorn processors.scene_captioning_processor:app --host 127.0.0.1 --port 8003 --log-level info >> "/home/runner/work/HackTemplate2/HackTemplate2/HackTemplate/logs/scene_captioning_processor.log" 2>&1
