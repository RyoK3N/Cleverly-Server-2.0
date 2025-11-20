.PHONY: install dev run test lint format clean docker-build docker-run help

# Default target
help:
	@echo "Cleverly Data Hub - Available Commands"
	@echo "======================================="
	@echo "make install     - Install production dependencies"
	@echo "make dev         - Install development dependencies"
	@echo "make run         - Run development server"
	@echo "make prod        - Run production server with gunicorn"
	@echo "make test        - Run test suite"
	@echo "make lint        - Run linters"
	@echo "make format      - Format code with black and isort"
	@echo "make clean       - Remove build artifacts"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run  - Run Docker container"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev,prod]"

# Running
run:
	PYTHONPATH=src python -m cleverly.app

prod:
	gunicorn -c gunicorn.conf.py wsgi:app

# Testing
test:
	pytest tests/ -v --cov=src/cleverly

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t cleverly:latest .

docker-run:
	docker-compose up -d

# Database/Data
init-data:
	mkdir -p data logs data/sessions

# Pipeline
pipeline:
	PYTHONPATH=src python -c "from cleverly.services.pipeline.cold_email import main; main()"
