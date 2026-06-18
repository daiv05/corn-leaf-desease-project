ifeq ($(OS),Windows_NT)
    PYTHON := venv\Scripts\python
    PIP    := venv\Scripts\pip
    RUFF   := venv\Scripts\ruff
else
    PYTHON := venv/bin/python
    PIP    := venv/bin/pip
    RUFF   := venv/bin/ruff
endif

.PHONY: install splits train test-loader summary lint fmt

install:
	$(PIP) install -e ".[dev]"

splits:
	$(PYTHON) scripts/create_splits.py

train:
	$(PYTHON) scripts/train.py

test-loader:
	$(PYTHON) test_loader_integration.py

summary:
	$(PYTHON) src/data/dataset_summary.py

lint:
	$(RUFF) check src/ scripts/ test_loader_integration.py

fmt:
	$(RUFF) format src/ scripts/ test_loader_integration.py
