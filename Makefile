# manwe — common tasks. Python training ground lives in ./python.
# (Recipes cd into python/ and use the checked-in uv lock.)
UV_RUN := uv run --locked --no-sync --

.PHONY: help setup setup-all test test-core lint fmt typecheck secret-scan smoke fusion-sim rust-build rust-check rust-bench-check clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install the numpy-only core + dev tools (editable)
	cd python && uv sync --locked --extra dev

setup-all: ## Install everything incl. heavy pillars (use Python 3.11–3.12)
	cd python && uv sync --locked --extra all --extra dev

test: ## Run the full pytest suite
	cd python && PYTHONWARNINGS=error $(UV_RUN) .venv/bin/python -m pytest tests

test-core: ## Run the same core/config suite in the prepared development environment
	cd python && PYTHONWARNINGS=error $(UV_RUN) .venv/bin/python -m pytest tests

lint: ## Ruff lint
	cd python && $(UV_RUN) .venv/bin/ruff check src tests

fmt: ## Ruff format
	cd python && $(UV_RUN) .venv/bin/ruff format src tests

typecheck: ## mypy on the package
	cd python && $(UV_RUN) .venv/bin/python -m mypy src/manwe

secret-scan: ## Scan current Git files for common credential material
	python3 scripts/check_secrets.py

smoke: ## Generate an offline synthetic dataset
	tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	  cd python && $(UV_RUN) .venv/bin/python -m manwe.cli synth "$$tmp/dataset"

fusion-sim: ## Compare all filters on a synthetic multi-sensor scenario
	cd python && $(UV_RUN) .venv/bin/python -m manwe.cli fusion-sim

rust-build: ## Build the Rust inference CLI (CPU)
	cargo build --release --locked --no-default-features

rust-check: ## Rust format check + clippy
	cargo fmt --all --check
	cargo fetch --locked
	cargo clippy --locked --offline --all-targets --no-default-features -- -D warnings
	cargo test --locked --offline --all-targets --no-default-features

rust-bench-check: ## Check the Apple-Metal benchmark crate (requires macOS/Metal)
	cd metal-yolo-tests && cargo fmt --all --check
	cd metal-yolo-tests && cargo fetch --locked
	cd metal-yolo-tests && cargo test --locked --offline --all-targets
	cd metal-yolo-tests && cargo clippy --locked --offline --all-targets -- -D warnings

clean: ## Remove local caches and build artifacts
	rm -rf target metal-yolo-tests/target python/dist python/.wheel-venv
	rm -rf python/.pytest_cache python/.ruff_cache python/.mypy_cache
	find python -name __pycache__ -type d -prune -exec rm -rf {} +
