PYTHON ?= python3.12

.PHONY: install dev test lint docker-up docker-down

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r apps/api/requirements.txt

dev:
	. .venv/bin/activate && uvicorn app.main:app --app-dir apps/api --reload --host 127.0.0.1 --port 8091

test:
	. .venv/bin/activate && pytest apps/api/tests

lint:
	. .venv/bin/activate && python -m compileall apps/api/app

docker-up:
	docker compose up --build

docker-down:
	docker compose down
