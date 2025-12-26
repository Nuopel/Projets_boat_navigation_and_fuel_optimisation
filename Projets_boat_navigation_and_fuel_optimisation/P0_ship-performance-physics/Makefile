.PHONY: help install install-dev test test-cov lint format type-check clean docs

help:
	@echo "Ship Performance Model - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package in production mode"
	@echo "  make install-dev   Install package with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run flake8 linter"
	@echo "  make format        Format code with black and isort"
	@echo "  make type-check    Run mypy type checker"
	@echo "  make check-all     Run all quality checks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove build artifacts and cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,viz,notebooks]"
	pre-commit install

test:
	pytest -v

test-cov:
	pytest --cov=ship_performance --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 src/ tests/

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

type-check:
	mypy src/

check-all: format lint type-check test
	@echo "✅ All checks passed!"

docs:
	cd docs && make html
	@echo "Documentation built: docs/_build/html/index.html"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
