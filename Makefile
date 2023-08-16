SRC_DIR := src

venv:
	poetry shell

install:
	poetry install

lint:
	poetry run flake8 $(SRC_DIR)/*.py

test:
	poetry run coverage run -m pytest src/test_*.py
	poetry run coverage report --show-missing

