.PHONY: setup train serve test docker-build docker-run

setup:
	pip install -r requirements-prod.txt
	pip install -r requirements-dev.txt

train:
	python -m entrypoint.train

serve:
	python -m entrypoint.inference

test:
	pytest tests/

docker-build:
	docker build -t card-classifier .

docker-run:
	docker run -p 8000:8000 card-classifier
