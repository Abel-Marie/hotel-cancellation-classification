# One command per workflow. Run `make help` for the list.
PY ?= python
URL ?=

.PHONY: help setup lint test test-all pipeline fast notebooks notebooks-fast standalone sync-service serve monitor check-live bundle-size all clean

help:
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | sed "s/:.*## /\t/"

setup: ## install the package and development tools
	$(PY) -m pip install -e ".[dev]"

lint: ## ruff checks
	ruff check src scripts tests service --exclude service/core/features.py

test: ## fast tests (unit, API, parity); skips what needs a finished run
	$(PY) -m pytest -m "not slow" -q

test-all: ## every test, including the end-to-end fast pipeline run
	$(PY) -m pytest -q

pipeline: ## the full pipeline: every stage, full budgets (about 20 minutes)
	hotel-cancellation all

fast: ## smoke run with tiny budgets (never for reported results)
	hotel-cancellation all --fast

notebooks: ## build and execute the notebook, full budgets
	$(PY) scripts/build_notebooks.py

notebooks-fast: ## build and execute the notebook in fast mode
	$(PY) scripts/build_notebooks.py --fast

standalone: ## prove the notebook runs alone in an empty folder
	$(PY) scripts/check_standalone.py

sync-service: ## copy feature code and exported model into service/
	$(PY) scripts/sync_service.py

serve: ## run the API locally on http://127.0.0.1:8000
	cd service && uvicorn main:app --reload --port 8000

monitor: ## drift report, plus latency if URL=https://... is given
	hotel-cancellation monitor drift
	@if [ -n "$(URL)" ]; then hotel-cancellation monitor latency --url $(URL); fi

check-live: ## test a deployed copy from the outside: make check-live URL=https://...
	$(PY) scripts/check_live.py $(URL)

bundle-size: ## size of the service bundle as Vercel would install it
	$(PY) scripts/bundle_size.py

all: pipeline sync-service notebooks test ## reproduce everything from a fresh clone

clean: ## remove generated data and build folders (keeps reports and the service model)
	rm -rf data/raw data/interim data/processed build .pytest_cache
