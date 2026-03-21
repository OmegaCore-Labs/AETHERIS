.PHONY: install install-dev install-cloud test lint format clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-cloud:
	pip install -e ".[cloud]"

test:
	pytest tests/ -v --cov=aetheris --cov-report=term-missing

test-parallel:
	pytest tests/ -n auto -v

lint:
	ruff check aetheris/
	mypy aetheris/ --ignore-missing-imports

format:
	black aetheris/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	pip install build
	python -m build

release: build
	twine upload dist/*

help:
	@echo "Available commands:"
	@echo "  make install       - Install AETHERIS"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make install-cloud - Install with cloud dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make build         - Build distribution"
