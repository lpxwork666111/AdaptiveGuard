#!/usr/bin/env bash
set -euo pipefail
python -m adaptiveguard.cli.main check-env
pytest -q
