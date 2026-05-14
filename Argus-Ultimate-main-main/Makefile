# Argus Ultimate — Makefile
# Push 81: consolidated targets

.PHONY: help install dev-install lint fmt typecheck security test test-core test-integration \
        docker-build docker-up docker-down docker-logs health validate-config \
        paper live backtest clean

PYTHON := python3
PIP    := pip3
DOCKER := docker
COMPOSE := docker-compose

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Dependencies ──────────────────────────────────────────────────────────
install:  ## Install production dependencies
	$(PIP) install -r requirements.txt

dev-install:  ## Install dev + test dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-mock httpx ruff black mypy bandit pip-audit

# ── Code Quality ──────────────────────────────────────────────────────────
lint:  ## Run ruff linter on core/
	ruff check core/ strategies/ execution/ risk/ backtest/ --output-format=github

fmt:  ## Auto-format with black + ruff --fix
	black core/ strategies/ execution/ risk/ backtest/
	ruff check core/ strategies/ execution/ risk/ backtest/ --fix

typecheck:  ## Run mypy on core/
	mypy core/ --ignore-missing-imports --no-error-summary

security:  ## Run bandit + pip-audit
	bandit -r core/ -ll || true
	pip-audit -r requirements.txt || true

# ── Tests ─────────────────────────────────────────────────────────────────
test:  ## Run all tests
	pytest tests/ tests_unified/ -q --tb=short \
	  --ignore=tests/live --ignore=tests/exchange_live \
	  -e ARGUS_MODE=test -e ARGUS_EXCHANGE=paper

test-core:  ## Run core/ tests only
	pytest tests/ -q --tb=short --cov=core --cov-report=term-missing \
	  -e ARGUS_MODE=test -e ARGUS_EXCHANGE=paper

test-integration:  ## Run integration smoke
	pytest tests_unified/ -q --tb=short \
	  -e ARGUS_MODE=test -e ARGUS_EXCHANGE=paper

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	ptw tests/ -- -q --tb=short

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:  ## Build production Docker image
	$(DOCKER) build -t argus-ultimate:latest .

docker-up:  ## Start full stack (bot + prometheus + grafana + redis)
	$(COMPOSE) up -d
	docker-up-wait: docker-up
	@echo "Waiting for services..."
	@sleep 5
	@$(MAKE) health

docker-down:  ## Stop all services
	$(COMPOSE) down

docker-logs:  ## Tail logs
	$(COMPOSE) logs -f argus-bot

docker-restart:  ## Restart argus-bot only
	$(COMPOSE) restart argus-bot

# ── Health & Validation ───────────────────────────────────────────────────
health:  ## Check running instance health
	$(PYTHON) scripts/health_check.py

validate-config:  ## Validate config file
	$(PYTHON) scripts/validate_config.py

# ── Run Modes ─────────────────────────────────────────────────────────────
paper:  ## Start paper trading (local)
	$(PYTHON) run_paper.py

live:  ## Start live trading (requires API keys in env)
	@$(MAKE) validate-config
	$(PYTHON) scripts/start.py

backtest:  ## Run backtest
	$(PYTHON) main.py --mode backtest

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml bandit-report.json pip-audit-report.json

clean-docker:  ## Remove Docker containers, volumes
	$(COMPOSE) down -v --remove-orphans
	$(DOCKER) rmi argus-ultimate:latest 2>/dev/null || true
