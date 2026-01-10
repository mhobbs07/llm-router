.PHONY: help build dev prod stop logs shell test clean compile deps deploy webui-logs

# Variables
IMAGE_NAME := llm-router
VERSION ?= latest
REGISTRY ?= docker.io/yourname

# Default target
help:
	@echo "LLM Router & Open WebUI - Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Run both services in development mode"
	@echo "  make shell        - Open shell in LLM Router container"
	@echo "  make logs         - Follow LLM Router logs"
	@echo "  make webui-logs   - Follow Open WebUI logs"
	@echo ""
	@echo "Production:"
	@echo "  make build        - Build LLM Router production image"
	@echo "  make prod         - Run both services in production mode"
	@echo "  make test         - Run tests in LLM Router container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make compile      - Compile requirements.txt from requirements.in"
	@echo "  make deps         - Sync local dependencies"
	@echo "  make clean        - Remove all containers and images"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy       - Build and push LLM Router to registry"

# Development
dev:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
build:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME):$(VERSION) .

prod:
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 \
	docker-compose up -d

# Utilities
stop:
	docker-compose down

logs:
	docker-compose logs -f llm-router

webui-logs:
	docker-compose logs -f open-webui

shell:
	docker-compose exec llm-router bash

test:
	docker-compose exec llm-router pytest tests/ -v

# Dependency management
compile:
	pip-compile requirements.in

deps: compile
	pip-sync requirements.txt

# Cleanup
clean:
	docker-compose down -v
	docker rmi $(IMAGE_NAME):$(VERSION) 2>/dev/null || true

# Deployment
deploy: build
	docker tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest

# Health check
health:
	@curl -sf http://localhost:8000/health || (echo "Health check failed" && exit 1)