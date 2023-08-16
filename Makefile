SRC_DIR := src

venv:
	poetry shell

install:
	poetry install

lint:
	poetry run flake8 $(SRC_DIR)/*.py

test:
	poetry run pytest --cov=. src/test_*.py

test-ci:
	poetry run pytest --cov=. --cov-report=xml src/test_*.py