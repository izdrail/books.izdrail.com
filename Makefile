#!/bin/sh -l
# Define variables
IMAGE_PROD=izdrail/book.izdrail.com:latest
DOCKERFILE=Dockerfile
DOCKER_COMPOSE_FILE=docker-compose.yaml
DOCKER_COMPOSE_FILE_PROD=docker-compose.yaml
CODE=""

# Check if buildx is available
BUILDX_AVAILABLE := $(shell docker buildx version >/dev/null 2>&1 && echo true || echo false)

build:
	docker image rm -f $(IMAGE_PROD) || true
ifeq ($(BUILDX_AVAILABLE),true)
	docker buildx build \
		--platform linux/amd64 \
		-t $(IMAGE_PROD) \
		--no-cache \
		--progress=plain \
		--build-arg CACHEBUST=$$(date +%s) \
		-f $(DOCKERFILE) \
		.
else
	docker build \
		-t $(IMAGE_PROD) \
		--no-cache \
		--build-arg CACHEBUST=$$(date +%s) \
		-f $(DOCKERFILE) \
		.
endif
dev:
	docker-compose -f $(DOCKER_COMPOSE_FILE_PROD) up --remove-orphans
prod:
	docker-compose -f $(DOCKER_COMPOSE_FILE_PROD) up --remove-orphans
down:
	docker-compose -f $(DOCKER_COMPOSE_FILE) down
ssh:
	docker exec -it book.izdrail.com /bin/bash