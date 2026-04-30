python-src := "packages/fedotllm/src packages/shared/src apps/frontend/src apps/server/src"
python-tests := "packages/fedotllm/tests"

# Sync the Python workspace
install:
    uv sync --all-packages

# Refresh the workspace lockfile and environment
update:
    uv sync --all-packages --upgrade



# Run ruff linter
lint-check:
    uv run ruff check {{python-src}} {{python-tests}}

# Auto-fix lint issues
lint:
    uv run ruff check --fix {{python-src}} {{python-tests}}

# Run ruff formatter (check only)
fmt-check:
    uv run ruff format --check {{python-src}} {{python-tests}}

# Run ruff formatter (write changes)
fmt:
    uv run ruff format {{python-src}} {{python-tests}}

# Run ty type checker
ty:
    uv run ty check {{python-src}} {{python-tests}}

# Run the Python test suite
test *FLAGS:
    uv run pytest -v {{FLAGS}}

# Run all quality
quality:
    just lint
    just fmt
    just ty
    just test

# Remove Python and frontend build artifacts
clean:
    rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache .ty_cache htmlcov
    find packages apps -type d -name __pycache__ -prune -exec rm -rf {} +
    find packages apps -name '*.pyc' -delete
    find packages -type d -name '*.egg-info' -prune -exec rm -rf {} +