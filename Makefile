.PHONY: install install-dev test lint smoke clean help

PYTHON := python
PIP    := pip

help:
	@echo "Available targets:"
	@echo "  install      Install package in editable mode"
	@echo "  install-dev  Install with dev extras"
	@echo "  test         Run all tests with pytest"
	@echo "  lint         Run ruff linter"
	@echo "  smoke        Run the synthetic smoke test"
	@echo "  clean        Remove build artifacts and caches"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	$(PYTHON) scripts/train.py \
	  --config configs/experiments/smoke_test.yaml \
	  --paths configs/paths/local.yaml

lint:
	ruff check src/ scripts/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/ .mypy_cache/
