# LeanVibe Agent Hive 2.0 - Development Makefile
# Provides common development commands and shortcuts

.PHONY: help setup setup-minimal setup-full setup-devcontainer install install-pre-commit \
    dev start start-minimal start-full start-bg stop restart \
    sandbox sandbox-demo sandbox-auto sandbox-showcase \
    test test-unit test-integration test-performance test-security test-e2e test-smoke test-cov test-fast test-integration-pytest test-watch \
    test-core-fast test-backend-fast test-prompt \
    lint format check security benchmark load-test \
	migrate rollback db-shell redis-shell monitor dev-tools \
	logs ps shell build clean docs health status verify-core \
	frontend-install frontend-dev frontend-build frontend-test \
	pwa-dev pwa-build pwa-generate-schemas \
	docs-generate-nav docs-validate-links \
	pre-commit ci release dev-container emergency-reset env-info

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Configuration
PYTHON := python3
PIP := pip
PYTEST := pytest
VENV_DIR := venv
COMPOSE := docker compose

# Self-documenting help target (robust implementation)
help: ## Show this help message
	@echo "$(BLUE)LeanVibe Agent Hive 2.0 - Development Commands$(NC)"
	@echo "==============================================="
	@echo ""
	@echo "$(YELLOW)Setup & Environment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'setup|install|env' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' || true
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'dev|start|stop|restart|sandbox' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' || true
	@echo ""
	@echo "$(YELLOW)Testing & Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'test|lint|format|check|security|benchmark' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' || true
	@echo ""
	@echo "$(YELLOW)Database & Services:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'db|redis|migrate|monitor' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' || true
	@echo ""
	@echo "$(YELLOW)Utilities & Tools:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE 'setup|install|env|dev|start|stop|restart|sandbox|test|lint|format|check|security|benchmark|db|redis|migrate|monitor' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' || true
	@echo ""
	@echo "$(BLUE)Quick Start:$(NC)"
	@echo "  $(GREEN)make setup$(NC)           # One-command setup"
	@echo "  $(GREEN)make start$(NC)           # Start all services" 
	@echo "  $(GREEN)make test$(NC)            # Run comprehensive tests"
	@echo "  $(GREEN)make health$(NC)          # Check system health"

# Setup & Environment
setup: ## Run full system setup (one-command setup)
	@echo "$(BLUE)🚀 Running full system setup...$(NC)"
	@./scripts/setup.sh fast

setup-minimal: ## Run minimal setup for CI/CD environments
	@echo "$(BLUE)⚡ Running minimal setup...$(NC)"
	@./scripts/setup.sh minimal

setup-full: ## Run complete setup with all tools
	@echo "$(BLUE)🔧 Running full setup...$(NC)"
	@./scripts/setup.sh full

setup-devcontainer: ## Run devcontainer setup
	@echo "$(BLUE)📦 Running devcontainer setup...$(NC)"
	@./scripts/setup.sh devcontainer

install: ## Install Python dependencies in virtual environment
	@echo "$(BLUE)📦 Installing Python dependencies...$(NC)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@. $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip setuptools wheel && \
		$(PIP) install -e .[dev,monitoring,ai-extended]
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

install-pre-commit: install ## Install pre-commit hooks
	@echo "$(BLUE)🔧 Installing pre-commit hooks...$(NC)"
	@. $(VENV_DIR)/bin/activate && pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks installed$(NC)"

# Development
dev: ## Start development server with auto-reload
	@echo "$(BLUE)🌟 Starting development server...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

start: ## Start all services (infrastructure + application)
	@echo "$(BLUE)🚀 Starting all services...$(NC)"
	@./scripts/start.sh fast

start-minimal: ## Start minimal services for CI/CD
	@echo "$(BLUE)⚡ Starting minimal services...$(NC)"
	@./scripts/start.sh minimal

start-full: ## Start all services including monitoring
	@echo "$(BLUE)🔧 Starting full services...$(NC)"
	@./scripts/start.sh full

start-bg: ## Start services in background
	@echo "$(BLUE)📦 Starting services in background...$(NC)"
	@BACKGROUND=true ./scripts/start.sh fast

stop: ## Stop all services
	@echo "$(BLUE)🛑 Stopping all services...$(NC)"
	@$(COMPOSE) down
	@echo "$(GREEN)✅ All services stopped$(NC)"

restart: stop start-bg ## Restart all services

# Sandbox & Demonstrations
sandbox: ## Start interactive sandbox mode
	@echo "$(BLUE)🎮 Starting interactive sandbox...$(NC)"
	@./scripts/sandbox.sh interactive

sandbox-demo: ## Run automated demo mode
	@echo "$(BLUE)🎬 Starting demo mode...$(NC)"
	@./scripts/sandbox.sh demo

sandbox-auto: ## Run autonomous development showcase
	@echo "$(BLUE)🤖 Starting autonomous showcase...$(NC)"
	@./scripts/sandbox.sh auto

sandbox-showcase: ## Run best-of showcase
	@echo "$(BLUE)🏆 Starting showcase mode...$(NC)"
	@./scripts/sandbox.sh showcase

# Testing & Quality
test: ## Run all tests
	@echo "$(BLUE)🧪 Running comprehensive test suite...$(NC)"
	@./scripts/test.sh all

test-unit: ## Run unit tests only
	@echo "$(BLUE)⚡ Running unit tests...$(NC)"
	@./scripts/test.sh unit

test-integration: ## Run integration tests
	@echo "$(BLUE)🔗 Running integration tests...$(NC)"
	@./scripts/test.sh integration

test-performance: ## Run performance tests
	@echo "$(BLUE)🚀 Running performance tests...$(NC)"
	@./scripts/test.sh performance

test-security: ## Run security tests
	@echo "$(BLUE)🔒 Running security tests...$(NC)"
	@./scripts/test.sh security

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)🌐 Running end-to-end tests...$(NC)"
	@./scripts/test.sh e2e

test-smoke: ## Run smoke tests
	@echo "$(BLUE)💨 Running smoke tests...$(NC)"
	@./scripts/test.sh smoke

test-cov: ## Run tests with coverage report (legacy)
	@echo "$(BLUE)🧪 Running tests with coverage...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTEST) --cov=app --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Focused lanes (fast feedback in CI/PRs)
test-core-fast: ## Fast lane: smoke + WS + prompt optimization core
	@echo "$(BLUE)⚡ Running core fast lane (smoke + ws + prompt)...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTEST) -q tests/smoke tests/ws tests/test_prompt_optimization_comprehensive.py

test-backend-fast: ## Fast lane: backend core modules (contracts + core + smoke)
	@echo "$(BLUE)⚡ Running backend fast lane...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTEST) -q tests/contracts tests/core tests/smoke -k "not dashboard"

test-prompt: ## Prompt optimization and related engines
	@echo "$(BLUE)🧠 Running prompt optimization test lane...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		$(PYTEST) -q tests/test_prompt_optimization_* tests/performance/test_prompt_optimization_benchmarks.py -k "not slow"

test-fast: ## Run tests without slow/integration tests
	@echo "$(BLUE)⚡ Running fast tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTEST) -v -m "not slow and not integration"

test-integration-pytest: ## Run integration tests with pytest (legacy method)
	@echo "$(BLUE)🔗 Running integration tests with pytest...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTEST) -v -m integration

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)👀 Running tests in watch mode...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTEST) -f

lint: ## Run code quality checks (ruff, black, mypy)
	@echo "$(BLUE)🔍 Running code quality checks...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		ruff check . && \
		black --check . && \
		mypy app
	@echo "$(GREEN)✅ Code quality checks passed$(NC)"

format: ## Format code with black and fix ruff issues
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		black . && \
		ruff --fix .
	@echo "$(GREEN)✅ Code formatted$(NC)"

check: lint test ## Run all quality checks (lint + test)

security: ## Run security checks
	@echo "$(BLUE)🔒 Running security checks...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		bandit -r app && \
		safety check
	@echo "$(GREEN)✅ Security checks passed$(NC)"

# Database & Services
migrate: ## Run database migrations
	@echo "$(BLUE)🗄️  Running database migrations...$(NC)"
	@. $(VENV_DIR)/bin/activate && alembic upgrade head
	@echo "$(GREEN)✅ Migrations completed$(NC)"

rollback: ## Rollback database migration
	@echo "$(BLUE)⏪ Rolling back database migration...$(NC)"
	@. $(VENV_DIR)/bin/activate && alembic downgrade -1
	@echo "$(GREEN)✅ Rollback completed$(NC)"

db-shell: ## Open PostgreSQL shell
	@echo "$(BLUE)🗄️  Opening database shell...$(NC)"
	@$(COMPOSE) exec postgres psql -U leanvibe_user -d leanvibe_agent_hive

redis-shell: ## Open Redis shell
	@echo "$(BLUE)📦 Opening Redis shell...$(NC)"
	@$(COMPOSE) exec redis redis-cli

# Utilities
health: ## Run comprehensive health check
	@echo "$(BLUE)🏥 Running health check...$(NC)"
	@./scripts/health.sh

logs: ## Show logs from all services
	@$(COMPOSE) logs -f

ps: ## Show running services
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

shell: ## Open shell in development environment
	@echo "$(BLUE)🐚 Opening development shell...$(NC)"
	@. $(VENV_DIR)/bin/activate && /bin/bash

build: ## Build Docker images
	@echo "$(BLUE)🏗️  Building Docker images...$(NC)"
	@$(COMPOSE) build
	@echo "$(GREEN)✅ Images built$(NC)"

clean: ## Clean up temporary files and containers
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@$(COMPOSE) down -v --remove-orphans
	@docker system prune -f
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage .mypy_cache/
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

docs: ## Generate and serve documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		mkdocs serve --dev-addr 0.0.0.0:8080
	@echo "$(GREEN)📖 Documentation available at http://localhost:8080$(NC)"

# Docs helpers
docs-generate-nav: ## Regenerate docs/NAV_INDEX.md
	@echo "$(BLUE)📚 Generating navigation index...$(NC)"
	@python3 scripts/docs/generate_nav_index.py
	@echo "$(GREEN)✅ Navigation index updated$(NC)"

docs-validate-links: ## Validate common doc links point to canonical targets
	@echo "$(BLUE)🔗 Validating docs links...$(NC)"
	@rg -n "SYSTEM_ARCHITECTURE.md|API_REFERENCE_COMPREHENSIVE.md|DOCS_INDEX.md" docs integrations mobile-pwa || true
	@echo "$(YELLOW)Review the above for outdated links (expect ARCHITECTURE.md, reference/, NAV_INDEX.md).$(NC)"

# Performance & Monitoring
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)⚡ Running performance benchmarks...$(NC)"
	@. $(VENV_DIR)/bin/activate && $(PYTEST) -v -m benchmark

load-test: ## Run load tests
	@echo "$(BLUE)🔥 Running load tests...$(NC)"
	@. $(VENV_DIR)/bin/activate && \
		locust -f tests/performance/locust_load_tests.py --headless -u 10 -r 2 -t 60s --host=http://localhost:8000

monitor: ## Start monitoring services (Prometheus + Grafana)
	@echo "$(BLUE)📊 Starting monitoring services...$(NC)"
	@$(COMPOSE) --profile monitoring up -d prometheus grafana
	@echo "$(GREEN)📈 Prometheus: http://localhost:9090$(NC)"
	@echo "$(GREEN)📊 Grafana: http://localhost:3001$(NC)"

# Development Tools
dev-tools: ## Start development tools (pgAdmin, Redis Insight, etc.)
	@echo "$(BLUE)🛠️  Starting development tools...$(NC)"
	@$(COMPOSE) --profile development up -d pgadmin redis-insight jupyter
	@echo "$(GREEN)🗄️  pgAdmin: http://localhost:5050$(NC)"
	@echo "$(GREEN)📦 Redis Insight: http://localhost:8001$(NC)"
	@echo "$(GREEN)📓 Jupyter: http://localhost:8888$(NC)"

# Frontend Development
frontend-install: ## Install frontend dependencies
	@echo "$(BLUE)🎨 Installing frontend dependencies...$(NC)"
	@cd frontend && npm install
	@echo "$(GREEN)✅ Frontend dependencies installed$(NC)"

frontend-dev: ## Start frontend development server
	@echo "$(BLUE)🎨 Starting frontend development server...$(NC)"
	@cd frontend && npm run dev

frontend-build: ## Build frontend for production
	@echo "$(BLUE)🏗️  Building frontend...$(NC)"
	@cd frontend && npm run build
	@echo "$(GREEN)✅ Frontend built$(NC)"

frontend-test: ## Run frontend tests
	@echo "$(BLUE)🧪 Running frontend tests...$(NC)"
	@cd frontend && npm run test

# Mobile PWA
pwa-dev: ## Start PWA development server
	@echo "$(BLUE)📱 Starting PWA development server...$(NC)"
	@cd mobile-pwa && npm run dev

pwa-build: ## Build PWA for production
	@echo "$(BLUE)🏗️  Building PWA...$(NC)"
	@cd mobile-pwa && npm run build

pwa-generate-schemas: ## Generate TS types from JSON schemas for PWA
	@echo "$(BLUE)🧬 Generating PWA schema types...$(NC)"
	@cd mobile-pwa && npm run -s generate:schemas

# CI/CD
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)🔧 Running pre-commit hooks...$(NC)"
	@. $(VENV_DIR)/bin/activate && pre-commit run --all-files

ci: install-pre-commit lint test security ## Run full CI pipeline locally

# Release
release: ## Create a new release (semantic versioning)
	@echo "$(BLUE)🚀 Creating release...$(NC)"
	@. $(VENV_DIR)/bin/activate && cz bump --changelog
	@echo "$(GREEN)✅ Release created$(NC)"

# Quick status check
status: ## Show quick system status
	@echo "$(BLUE)📊 System Status$(NC)"
	@echo "==============="
	@echo "$(YELLOW)Docker Services:$(NC)"
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "Docker Compose not available"
	@echo ""
	@echo "$(YELLOW)API Health:$(NC)"
	@curl -s http://localhost:8000/health | jq -r '.status // "API not responding"' 2>/dev/null || echo "API not responding"
	@echo ""
	@echo "$(YELLOW)Virtual Environment:$(NC)"
	@if [ -d "$(VENV_DIR)" ]; then echo "✅ Virtual environment exists"; else echo "❌ Virtual environment missing"; fi

# Core verification (REST + WS)
verify-core: ## Verify core endpoints (health, metrics, WebSocket handshake)
	@echo "$(BLUE)🔎 Verifying core API/WS...$(NC)"
	@python3 scripts/verify_core.py

# Development container support
dev-container: ## Open project in VS Code dev container
	@echo "$(BLUE)📦 Opening in VS Code dev container...$(NC)"
	@code --folder-uri vscode-remote://dev-container+$(shell pwd | sed 's/\//%2F/g')/workspace

# Emergency/Recovery
emergency-reset: ## Emergency reset - stop everything and clean up
	@echo "$(RED)🚨 Emergency reset - stopping all services...$(NC)"
	@$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	@docker container prune -f 2>/dev/null || true
	@docker volume prune -f 2>/dev/null || true
	@echo "$(GREEN)✅ Emergency reset completed$(NC)"
	@echo "$(YELLOW)⚠️  Run 'make setup' to reinitialize the system$(NC)"

# Show environment info
env-info: ## Show environment information
	@echo "$(BLUE)🌍 Environment Information$(NC)"
	@echo "=========================="
	@echo "$(YELLOW)Python:$(NC) $(shell python3 --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Docker:$(NC) $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Docker Compose:$(NC) $(shell docker compose version --short 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Git:$(NC) $(shell git --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Node.js:$(NC) $(shell node --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)Virtual Env:$(NC) $(if $(wildcard $(VENV_DIR)),✅ Exists,❌ Missing)"
	@echo "$(YELLOW)Config File:$(NC) $(if $(wildcard .env.local),✅ Exists,❌ Missing)"