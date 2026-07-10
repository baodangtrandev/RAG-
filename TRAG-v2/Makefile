.PHONY: install format lint test check all

install:
	pip install -e .
	pip install black isort flake8 pytest pytest-cov

format:
	black scripts src tests
	isort scripts src tests

lint:
	flake8 scripts src tests --max-line-length=120
	black --check scripts src tests
	isort --check-only scripts src tests

test:
	pytest tests/ --cov=scripts --cov=src --cov-report=term-missing

check: format lint test

all: check
