#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python scripts/gen_data.py --config configs/data.yaml --device cuda
