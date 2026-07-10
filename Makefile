# Upscale-BE Makefile
IMAGE ?= upscale-be:latest
COMPOSE ?= docker compose

.PHONY: help build up down restart logs shell test lint fmt clean pull-weights

help:
	@echo "Targets:"
	@echo "  build         Build Docker image"
	@echo "  up            Start stack (detached)"
	@echo "  down          Stop stack"
	@echo "  restart       Restart stack"
	@echo "  logs          Tail logs"
	@echo "  shell         Shell into api container"
	@echo "  test          Run pytest inside container"
	@echo "  lint          Run ruff check"
	@echo "  fmt           Run ruff format"
	@echo "  pull-weights  Download model weights"
	@echo "  clean         Prune dangling images/volumes"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f --tail=200

shell:
	$(COMPOSE) exec api bash

test:
	$(COMPOSE) run --rm api pytest -q

lint:
	$(COMPOSE) run --rm api ruff check app tests

fmt:
	$(COMPOSE) run --rm api ruff format app tests

pull-weights:
	mkdir -p weights
	curl -L -o weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

clean:
	docker system prune -f
