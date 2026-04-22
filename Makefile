# Makefile — AI Replication Sandbox development helpers

.PHONY: help install test lint coverage coverage-html coverage-report clean

PYTHON     ?= python3
PYTEST     ?= $(PYTHON) -m pytest
SRC        := src/replication
TESTS      := tests
COV_MIN    := 80

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package + dev dependencies
	pip install -r requirements-dev.txt
	pip install -e .

test: ## Run tests (no coverage)
	$(PYTEST) $(TESTS)/ -v --tb=short

lint: ## Run linters (flake8 + mypy)
	flake8 $(SRC)/ $(TESTS)/ --max-line-length 120 --extend-ignore E501
	mypy $(SRC)/ --ignore-missing-imports --no-error-summary || true

coverage: ## Run tests with coverage, enforce $(COV_MIN)% minimum
	$(PYTEST) $(TESTS)/ -v --tb=short \
		--cov=$(SRC) \
		--cov-branch \
		--cov-report=term-missing:skip-covered \
		--cov-report=json:coverage.json \
		--cov-fail-under=$(COV_MIN)

coverage-html: ## Run tests with HTML coverage report
	$(PYTEST) $(TESTS)/ -v --tb=short \
		--cov=$(SRC) \
		--cov-branch \
		--cov-report=html:htmlcov \
		--cov-report=term-missing:skip-covered
	@echo "\n  Open htmlcov/index.html to view the report"

coverage-report: ## Show coverage for files below $(COV_MIN)%
	@$(PYTEST) $(TESTS)/ -q \
		--cov=$(SRC) \
		--cov-branch \
		--cov-report=json:coverage.json \
		--no-header 2>/dev/null; \
	$(PYTHON) -c " \
import json, sys; \
d = json.load(open('coverage.json')); \
total = d['totals']['percent_covered']; \
print(f'\n  Total: {total:.1f}%\n'); \
low = [(f, s['summary']['percent_covered']) \
       for f, s in d['files'].items() \
       if s['summary']['percent_covered'] < $(COV_MIN)]; \
low.sort(key=lambda x: x[1]); \
if low: \
    print(f'  {len(low)} file(s) below $(COV_MIN)%:'); \
    [print(f'    {pct:5.1f}%  {f}') for f, pct in low]; \
else: \
    print('  All files meet $(COV_MIN)% threshold ✓'); \
"

coverage-xml: ## Generate Cobertura XML for CI upload
	$(PYTEST) $(TESTS)/ -q \
		--cov=$(SRC) \
		--cov-branch \
		--cov-report=xml:coverage.xml \
		--no-header

clean: ## Remove build artifacts and coverage data
	rm -rf htmlcov/ .coverage coverage.xml coverage.json .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
