# Makefile
# One-command startup for the Learning Accelerator.
#
# Usage:
#   make setup      : create/update the uv environment from uv.lock
#   make lock       : update uv.lock after changing pyproject.toml
#   make sync       : install the exact locked dependency versions
#   make gateway    : start the LiteLLM inference gateway
#   make run        : run the main application
#   make services   : start both A2A services in background
#   make stop       : stop background services
#   make langfuse   : start Langfuse observability stack
#   make test       : run fast unit tests
#   make eval       : run LLM-as-judge evaluation tests
#   make clean      : remove generated files

.PHONY: setup lock sync gateway gateway-stop gateway-logs postgres postgres-stop \
        postgres-logs run streamlit run-goal resume services stop langfuse \
        langfuse-stop langfuse-logs test test-postgres eval test-all clean \
        clean-all help

UV ?= uv
VENV = .venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest

# Setup

setup:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv is required but was not found on PATH."; \
		echo "Install it from https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}
	@echo "Syncing environment from uv.lock..."
	$(UV) sync --locked
	@cp -n .env.example .env 2>/dev/null || true
	@mkdir -p data
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. source $(VENV)/bin/activate"
	@echo "  2. Edit .env, point OLLAMA_BASE_URL at your Ollama endpoint"
	@echo "  3. Align the model in litellm_config.yaml with your pulled model"
	@echo "     (default: ollama pull qwen2.5:7b)"
	@echo "  4. make gateway && make run"

lock:
	@echo "Resolving dependencies and updating uv.lock..."
	$(UV) lock

sync:
	@echo "Syncing environment from uv.lock..."
	$(UV) sync --locked

# Inference gateway
# Every agent talks to this gateway; it routes to Ollama (see
# litellm_config.yaml). All LLM-backed commands need it running.
# The gateway is a docker container (see docker-compose.gateway.yml),
# so no litellm package needs to be installed in the venv.

gateway:
	@echo "Starting LiteLLM gateway (docker)..."
	docker compose -f docker-compose.gateway.yml up -d --wait
	@echo "Gateway up: http://localhost:4000 (logs: make gateway-logs)"

gateway-stop:
	docker compose -f docker-compose.gateway.yml down
	@echo "Gateway stopped"

gateway-logs:
	docker compose -f docker-compose.gateway.yml logs -f

# Checkpoint store (PostgreSQL)
# LangGraph checkpoints go to Postgres when CHECKPOINT_DB_URL /
# CHECKPOINT_DB_URL_UI are set in .env. Without them the app falls back
# to the legacy SQLite files (CHECKPOINT_DB). Independent of Langfuse.

postgres:
	@echo "Starting checkpoint PostgreSQL (:5433)..."
	docker compose -f docker-compose.checkpoint.yml up -d --wait
	@echo "Checkpoint Postgres up: localhost:5433 (logs: make postgres-logs)"

postgres-stop:
	docker compose -f docker-compose.checkpoint.yml down
	@echo "Checkpoint Postgres stopped (data volume preserved)"

postgres-logs:
	docker compose -f docker-compose.checkpoint.yml logs -f

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

test-postgres:
	@echo "Running Postgres checkpointer integration tests..."
	@docker compose -f docker-compose.checkpoint.yml up -d --wait
	TEST_CHECKPOINT_DB_URL=postgresql://notes:notes@localhost:5433/learning_accelerator \
	TEST_CHECKPOINT_DB_URL_UI=postgresql://notes:notes@localhost:5433/learning_accelerator_ui \
	$(PYTEST) tests/test_checkpointing.py -v

eval:
	@echo "Running evaluation tests (requires the LiteLLM gateway, ~90 seconds)..."
	$(PYTEST) tests/test_eval.py -v -s -m eval

test-all:
	@echo "Running all tests..."
	$(PYTEST) tests/ -v

# Maintenance

clean:
	@echo "Cleaning generated files..."
	@rm -f data/checkpoints.db data/checkpoints_ui.db
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete (venv preserved)"
	@echo "Note: Postgres-backed sessions are NOT removed. Stop them with"
	@echo "      'make postgres-stop' or drop the checkpoint_postgres_data volume."

clean-all: clean
	@rm -rf $(VENV)
	@echo "Full clean complete"

# Help

help:
	@echo "Learning Accelerator, available commands:"
	@echo ""
	@echo "  make setup        Sync the uv environment from uv.lock"
	@echo "  make lock         Update uv.lock after changing pyproject.toml"
	@echo "  make sync         Install exact locked dependency versions"
	@echo "  make gateway      Start the LiteLLM inference gateway (:4000, Docker)"
	@echo "  make gateway-stop Stop the gateway"
	@echo "  make gateway-logs Tail the gateway log"
	@echo "  make postgres     Start checkpoint PostgreSQL (:5433, Docker)"
	@echo "  make postgres-stop Stop checkpoint PostgreSQL"
	@echo "  make postgres-logs Tail checkpoint PostgreSQL logs"
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
	@echo "  make test-postgres Run Postgres checkpointer integration tests"
	@echo "  make eval         Run eval tests (~90 seconds, gateway required)"
	@echo ""
	@echo "  make clean        Remove SQLite checkpoints and cache files"
	@echo "                    (Postgres data survives; see make postgres-stop)"
