.PHONY: install test lint format typecheck text-check check

install:
	python -m pip install -e '.[dev,environments]'

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy adaptiveguard

text-check:
	python scripts/check_submission_text.py

check: lint typecheck text-check test
