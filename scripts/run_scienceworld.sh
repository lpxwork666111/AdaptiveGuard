#!/usr/bin/env bash
set -euo pipefail
python -m adaptiveguard.cli.main run --config configs/scienceworld.yaml "$@"
