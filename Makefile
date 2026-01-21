.PHONY: install dev run test lint typecheck

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy app
