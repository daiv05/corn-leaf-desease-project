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

.PHONY: install splits splits-sample train train-baselines train-baselines-full test-loader summary lint fmt

install:
	$(PIP) install -e ".[dev]"

splits:
	$(PYTHON) scripts/pipeline/create_splits.py

splits-sample:
	$(PYTHON) scripts/pipeline/create_splits.py --sample-fraction 0.2

train:
	$(PYTHON) scripts/pipeline/train.py

train-baselines:
	$(PYTHON) scripts/pipeline/train_baselines.py --models all \
		--splits-dir $${DATASET_ROOT}/splits/seed_42_sample20

train-baselines-full:
	$(PYTHON) scripts/pipeline/train_baselines.py --models all \
		--splits-dir $${DATASET_ROOT}/splits/seed_42

summary:
	$(PYTHON) src/data/dataset_summary.py

lint:
	$(RUFF) check src/ scripts/

lint-fix:
	$(RUFF) check --fix src/ scripts/

fmt:
	$(RUFF) format src/ scripts/

check:
	$(PYRIGHT) src/ scripts/
