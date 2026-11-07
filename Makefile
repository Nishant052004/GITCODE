.PHONY: install dev up down test lint ingest-demo query-demo eval eval-answer clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

dev:
	python -m uvicorn backend.app.main:app --reload --port 8000

up:
	docker compose up --build

down:
	docker compose down

test:
	pytest -q

lint:
	ruff check backend cli eval tests
	ruff format --check backend cli eval tests

format:
	ruff format backend cli eval tests

ingest-demo:
	python -m cli.main ingest examples/demo_repo

query-demo:
	python -m cli.main query "How do we verify a user's password?" --k 3

eval: eval-retrieval eval-answer

eval-retrieval:
	python eval/run_retrieval_eval.py --repo examples/demo_repo

eval-answer:
	python eval/run_answer_eval.py --repo examples/demo_repo

clean:
	rm -rf data .pytest_cache .ruff_cache
