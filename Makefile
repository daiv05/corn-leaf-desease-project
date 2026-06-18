PYTHON := venv/bin/python
PIP    := venv/bin/pip

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
	venv/bin/ruff check src/ scripts/ test_loader_integration.py

fmt:
	venv/bin/ruff format src/ scripts/ test_loader_integration.py
