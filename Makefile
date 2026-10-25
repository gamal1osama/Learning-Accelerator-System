# Makefile
# One-command startup for the Learning Accelerator.
#
# Usage:
#   make setup      : create venv and install dependencies
#   make gateway    : start the LiteLLM inference gateway
#   make run        : run the main application
#   make services   : start both A2A services in background
#   make stop       : stop background services
#   make langfuse   : start Langfuse observability stack
#   make test       : run fast unit tests
#   make eval       : run LLM-as-judge evaluation tests
#   make clean      : remove generated files

.PHONY: setup gateway gateway-stop gateway-logs run streamlit run-goal resume \
        services stop langfuse langfuse-stop langfuse-logs test eval test-all \
        clean clean-all help

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(PYTHON) -m pip
PYTEST = $(VENV)/bin/pytest

# Setup

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@cp -n .env.example .env 2>/dev/null || true
	@mkdir -p data
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. source $(VENV)/bin/activate"
	@echo "  2. Edit .env, point OLLAMA_BASE_URL at your Ollama endpoint"
	@echo "  3. Align the model in litellm_config.yaml with your pulled model"
	@echo "     (default: ollama pull qwen2.5:7b)"
	@echo "  4. make gateway && make run"

# Inference gateway
# Every agent talks to this gateway; it routes to Ollama (see
# litellm_config.yaml). All LLM-backed commands need it running.

gateway:
	@echo "Starting LiteLLM gateway..."
	@$(PYTHON) -c "import apscheduler" 2>/dev/null || $(PIP) install -q -r requirements.txt
	@mkdir -p logs
	@echo "Stopping existing gateway (if any)..."
	@pkill -f "[l]itellm.*litellm_config.yaml" 2>/dev/null || true
	@sleep 1
	# Load .env first (set -a exports loaded vars) so that .env overrides
	# any stale shell-exported variables the gateway would otherwise see.
	@set -a && . ./.env && set +a && $(VENV)/bin/litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000 --detailed_debug > logs/litellm_gateway.log 2>&1 &
	@sleep 3
	@for i in 1 2 3 4 5; do curl -sf http://127.0.0.1:4000/v1/models > /dev/null && break || sleep 2; done
	@curl -sf http://127.0.0.1:4000/v1/models > /dev/null \
		&& echo "Gateway up: http://localhost:4000 (logs: logs/litellm_gateway.log)" \
		|| echo "Gateway may still be booting - check logs/litellm_gateway.log"

gateway-stop:
	@pkill -f "[l]itellm.*litellm_config.yaml" 2>/dev/null || true
	@echo "Gateway stopped"

gateway-logs:
	@tail -f logs/litellm_gateway.log

# Running

run:
	@echo "Starting Learning Accelerator..."
	@curl -sf http://127.0.0.1:4000/v1/models > /dev/null 2>&1 \
		|| echo "WARNING: LiteLLM gateway not reachable on :4000. Start it with: make gateway"
	$(PYTHON) main.py

streamlit:
	@test -f streamlit_app.py || { \
		echo "streamlit_app.py not found in repo yet"; exit 1; }
	@echo "Starting Learning Accelerator (Streamlit UI)..."
	@echo "Open: http://localhost:8501"
	$(PYTHON) -m streamlit run streamlit_app.py --server.headless=true

run-goal:
	@echo "Starting with custom goal: $(GOAL)"
	$(PYTHON) main.py "$(GOAL)"

resume:
	@echo "Resuming session: $(SESSION)"
	$(PYTHON) main.py --resume $(SESSION)

# A2A Services

services: stop
	@echo "Starting A2A services..."
	@mkdir -p logs
	$(PYTHON) -u src/a2a_services/quiz_service.py > logs/quiz_service.log 2>&1 &
	@sleep 1
	$(PYTHON) -u src/crewai_agent/study_buddy.py > logs/study_buddy.log 2>&1 &
	@sleep 1
	@echo ""
	@echo "Services started:"
	@echo "  Quiz Generator:  http://localhost:9001/.well-known/agent-card.json"
	@echo "  CrewAI Study Buddy: http://localhost:9002/.well-known/agent-card.json"
	@echo "  (logs: logs/quiz_service.log, logs/study_buddy.log)"
	@echo ""
	@echo "Run 'make run' to start the main application"

stop:
	@echo "Stopping A2A services..."
	# [b]racket trick: prevents pkill from matching (and killing) its own
	# recipe shell, which otherwise aborts the target with 'Terminated'
	@pkill -f "[q]uiz_service.py" 2>/dev/null || true
	@pkill -f "[s]tudy_buddy.py" 2>/dev/null || true
	@echo "Services stopped"

# Observability

langfuse:
	@echo "Starting Langfuse..."
	docker compose up -d
	@echo ""
	@echo "Langfuse UI: http://localhost:3000"
	@echo "Create a project and add keys to .env"

langfuse-stop:
	docker compose down

langfuse-logs:
	docker compose logs -f langfuse-server

# Testing

test:
	@echo "Running unit tests..."
	$(PYTEST) tests/ -m "not eval" -v

eval:
	@echo "Running evaluation tests (requires the LiteLLM gateway, ~90 seconds)..."
	$(PYTEST) tests/test_eval.py -v -s -m eval

test-all:
	@echo "Running all tests..."
	$(PYTEST) tests/ -v

# Maintenance

clean:
	@echo "Cleaning generated files..."
	@rm -f data/checkpoints.db
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete (venv preserved)"

clean-all: clean
	@rm -rf $(VENV)
	@echo "Full clean complete"

# Help

help:
	@echo "Learning Accelerator, available commands:"
	@echo ""
	@echo "  make setup        Set up virtual environment and install deps"
	@echo "  make gateway      Start the LiteLLM inference gateway (:4000)"
	@echo "  make gateway-stop Stop the gateway"
	@echo "  make gateway-logs Tail the gateway log"
	@echo "  make run          Run the main application"
	@echo "  make run-goal GOAL='Learn async Python'  Run with custom goal"
	@echo "  make resume SESSION=abc12345  Resume a stopped session"
	@echo ""
	@echo "  make services     Start both A2A services in background"
	@echo "  make stop         Stop background services"
	@echo ""
	@echo "  make langfuse     Start Langfuse observability (Docker required)"
	@echo "  make langfuse-stop  Stop Langfuse"
	@echo ""
	@echo "  make test         Run fast unit tests (~3 seconds)"
	@echo "  make eval         Run eval tests (~90 seconds, gateway required)"
	@echo ""
	@echo "  make clean        Remove checkpoints and cache files"
