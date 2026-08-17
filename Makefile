.PHONY: setup run test docker-build docker-up docker-down evaluate clean

# On Windows, run the commands inside setup manually (or use WSL / Git Bash).
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

evaluate:
	python scripts/evaluate.py --dataset eval/questions.json --report eval/report.csv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
