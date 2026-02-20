.PHONY: init format lint

install:
	@uv sync 
	@echo
	@echo ------------------ PROJECT READY ---------------------- 
	@echo "Activate env: source .venv/bin/activate"
	@echo "Train model: make train"
	@echo

train:
	@python3 -m src.train

format:
	ruff src 

lint:
	ruff check src 