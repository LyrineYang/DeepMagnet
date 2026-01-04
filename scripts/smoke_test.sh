#!/usr/bin/env bash
set -e

# Ensure we're in project root
cd "$(dirname "$0")/.."

python scripts/gen_data.py --config configs/data_tiny.yaml --device cpu
python -m src.trainers.train --data configs/data_tiny.yaml --model configs/model.yaml --train configs/train_tiny.yaml
