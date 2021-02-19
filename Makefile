define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

#################################################################################
# COMMANDS                                                                      #
#################################################################################

help:  ## Print this help message.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-test  ## Remove all build, test, coverage and Python artifacts.

.PHONY: clean-build
clean-build:  ## Remove build artifacts.
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc:  ## Remove Python file artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:  ## Remove test and coverage artifacts.
	rm -f .coverage coverage.xml
	rm -fr htmlcov/ .junit/
	rm -fr .pytest_cache

.PHONY: env
env:  ## Create/update conda environment.
	conda env create --file environment.yml

.PHONY: black
black:  ## Format your code using black.
	python -m black --version
	python -m black --check .

.PHONY: lint
lint:  ## Lint your code using pylint.
	python -m pylint --version
	python -m pylint src

.PHONY: precommit
precommit:
	pre-commit run --all

.PHONY: test
test: clean-test clean-pyc  ## Run tests using pytest.
	python -m pytest --version
	python -m pytest tests --junitxml=.junit/test-results.xml \
		--cov=src/prophet_remade --cov-report=xml --cov-report=term

.PHONY: ci
ci: precommit lint test

.PHONY: docs
docs:  ## Generate Sphinx HTML documentation, including API docs.
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
