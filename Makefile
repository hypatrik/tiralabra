SRC_DIR := src

venv:
	poetry shell

install:
	poetry install

lint:
	poetry run flake8 $(SRC_DIR)/*.py

test:
	poetry run pytest src/*.py