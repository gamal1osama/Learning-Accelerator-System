# Makefile
# One-command startup for the Learning Accelerator.
#
# Usage:
#   make setup      : create venv and install dependencies
#   make run        : run the main application
#   make services   : start both A2A services in background
#   make stop       : stop background services
#   make langfuse   : start Langfuse observability stack
#   make test       : run fast unit tests
#   make eval       : run LLM-as-judge evaluation tests
#   make clean      : remove generated files

.PHONY: setup run streamlit run-goal resume services stop langfuse langfuse-stop \
        langfuse-logs test eval test-all clean clean-all help

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
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
	@echo "  2. Edit .env, set OLLAMA_MODEL to match your VRAM"
	@echo "  3. ollama pull qwen2.5:7b  (or qwen2.5-coder:32b for 24GB)"
	@echo "  4. make run"

# Running

run:
	@echo "Starting Learning Accelerator..."
	$(PYTHON) main.py

streamlit:
	@test -f streamlit_app.py || { \
		echo "streamlit_app.py not found in repo yet"; exit 1; }
	@echo "Starting Learning Accelerator (Streamlit UI)..."
	@echo "Open: http://localhost:8501"
	$(PYTHON) -m streamlit run streamlit_app.py

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
	@echo "Running evaluation tests (requires Ollama, ~90 seconds)..."
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
	@echo "  make run          Run the main application"
	@echo "  make run-goal GOAL='Learn async Python'  Run with custom goal"
	@echo "  make resume SESSION=abc12345  Resume a stopped session"
	@echo ""
	@echo "  make services     Start both A2A services in background"
	@echo "  make stop         Stop background A2A services"
	@echo ""
	@echo "  make langfuse     Start Langfuse observability (Docker required)"
	@echo "  make langfuse-stop  Stop Langfuse"
	@echo ""
	@echo "  make test         Run fast unit tests (~3 seconds)"
	@echo "  make eval         Run eval tests (~90 seconds, Ollama required)"
	@echo ""
	@echo "  make clean        Remove checkpoints and cache files"
