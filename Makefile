# Makefile for Claude-4.6 LLM Server

# Configuration
PYTHON_EXEC = ./venv/Scripts/python
PY = $(PYTHON_EXEC) -m
IMAGE_NAME = huihui-server
TAG = latest
PORT = 8000
HOST = 0.0.0.0
MODEL_CACHE_HOST = D:/llm-server/.cache/huggingface

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies from requirements.txt"
	@echo "  make run             - Start the FastAPI server locally"
	@echo "  make client          - Run the client script (standard)"
	@echo "  make client-stream   - Run the client script (streaming)"
	@echo "  make docker-build    - Build the Docker image"
	@echo "  make docker-run      - Run the Docker container (CPU)"
	@echo "  make docker-run-gpu  - Run the Docker container (GPU/NVIDIA)"
	@echo "  make docker-save     - Export Docker image to .tar.gz"
	@echo "  make docker-load     - Load Docker image from .tar.gz"

# --- Local Development ---

.PHONY: install
install:
	$(PY) pip install -r requirements.txt

.PHONY: run
run:
	$(PY) uvicorn app_llamacpp:app --host $(HOST) --port $(PORT) --workers 1

.PHONY: client
client:
	$(PYTHON_EXEC) client.py

.PHONY: client-stream
client-stream:
	$(PYTHON_EXEC) client.py --stream

# --- Docker Operations ---

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: docker-run
docker-run:
	docker run -d \
		--name huihui \
		-p $(PORT):$(PORT) \
		-v $(MODEL_CACHE_HOST):/app/model_cache \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)

.PHONY: docker-run-gpu
docker-run-gpu:
	docker run -d \
		--name huihui \
		--gpus all \
		-p $(PORT):$(PORT) \
		-v $(MODEL_CACHE_HOST):/app/model_cache \
		--restart unless-stopped \
		$(IMAGE_NAME):$(TAG)

.PHONY: docker-save
docker-save:
	docker save $(IMAGE_NAME):$(TAG) | gzip > $(IMAGE_NAME).tar.gz

.PHONY: docker-load
docker-load:
	docker load < $(IMAGE_NAME).tar.gz

# --- Testing ---

.PHONY: health
health:
	curl http://localhost:$(PORT)/health

.PHONY: test-chat
test-chat:
	curl -X POST http://localhost:$(PORT)/chat \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"Hello!"}]}'
