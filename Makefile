.PHONY: init format check requirements

init:
	python -m pip install -q -U poetry
	poetry install

format:
	poetry run isort --profile black -l 119 src
	poetry run ruff format src

check:
	poetry run ruff check src

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev
w

