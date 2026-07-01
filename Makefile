ifeq ($(OS),Windows_NT)
    PYTHON 	:= venv\Scripts\python
    PIP    	:= venv\Scripts\pip
    RUFF   	:= venv\Scripts\ruff
	PYRIGHT := venv\Scripts\pyright
else
    PYTHON 	:= venv/bin/python
    PIP    	:= venv/bin/pip
    RUFF   	:= venv/bin/ruff
	PYRIGHT := venv/bin/pyright
endif

MODELS ?= all

.PHONY: install download-dataset splits splits-baseline train train-baselines train-baselines-full test-loader summary docs-eda lint fmt

install:
	$(PIP) install -e ".[dev,analysis]"

download-dataset:
	$(PYTHON) scripts/dataset/download_dataset.py

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-baseline:
	$(PYTHON) scripts/pipeline/create_splits.py --baseline

train:
	$(PYTHON) scripts/pipeline/train.py

train-baselines:
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) \
		--splits-dir $${DATASET_ROOT}/splits/seed_42_baseline

train-baselines-full:
	$(PYTHON) scripts/pipeline/train_baselines.py --models $(MODELS) \
		--splits-dir $${DATASET_ROOT}/splits/seed_42

summary:
	$(PYTHON) src/analysis/dataset_summary.py

docs-eda:
	cp tmp/eda_*.png public/eda/

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/
