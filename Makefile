# Project configuration
UV = uv

# Detect Docker Compose command (prefer V2 plugin, fall back to V1 standalone)
DOCKER_COMPOSE := $(shell if docker compose version >/dev/null 2>&1; then echo "docker compose"; else echo "docker-compose"; fi)

# Colors for output
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
NC = \033[0m # No Color

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Project Commands$(NC)"
	@echo "$(BLUE)========================$(NC)"
	@echo "$(GREEN)Development Workflow:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { if ($$1 !~ /^docker-/ && $$1 !~ /^help$$/) printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: lint
lint: ## Run linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	${UV} run ruff check .

.PHONY: lint-fix
lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	${UV} run ruff check --fix .

.PHONY: format
format: ## Format code and organize imports with ruff
	@echo "$(BLUE)Formatting code and organizing imports...$(NC)"
	${UV} run ruff check --select I --fix .
	${UV} run ruff format .

.PHONY: quality
quality: format lint-fix ## Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(NC)"

# =============================================================================
# DOCKER
# =============================================================================

.PHONY: docker-up
docker-up: ## Start Docker containers with docker-compose
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d --build
	@echo "$(GREEN)Containers started!$(NC)"

.PHONY: docker-down
docker-down: ## Stop and remove Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Containers stopped!$(NC)"

.PHONY: docker-logs
docker-logs: ## View Docker container logs
	@echo "$(BLUE)Showing Docker logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

.PHONY: docker-shell
docker-shell: ## Open a shell inside the container
	@echo "$(BLUE)Connecting to container shell...$(NC)"
	$(DOCKER_COMPOSE) exec fedotllm bash

.PHONY: docker-exec
docker-exec: ## Execute a command in the container (use with CMD="<command>")
	@echo "$(BLUE)Executing command in container...$(NC)"
	$(DOCKER_COMPOSE) exec fedotllm $(CMD)