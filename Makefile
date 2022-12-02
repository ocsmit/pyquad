SHELL := bash
all: lint coverage

coverage:  ## Run tests with coverage
	poetry run coverage erase
	poetry run coverage run --include=tests/* -m pytest
	poetry run coverage report -m

lint:  ## Lint and static-check
	poetry run black spadmon --diff --line-length=80 --color 
	poetry run mypy spadmon

push:  ## Push code with tags
	git push && git push --tags

test:  ## Run tests
	poetry run pytest -ra


conda:
	poetry2conda pyproject.toml environment.yml
