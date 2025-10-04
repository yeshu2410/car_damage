.PHONY: setup data train-resnet train-yolo tune infer serve clean

PYTHON := python
PIP := pip
VENV := venv

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)\Scripts\activate && $(PIP) install --upgrade pip
	$(VENV)\Scripts\activate && $(PIP) install -r requirements.txt
	$(VENV)\Scripts\activate && $(PIP) install -e .[dev]
	$(VENV)\Scripts\activate && pre-commit install

data:
	$(VENV)\Scripts\activate && $(PYTHON) -m src.data.pipeline

train-resnet:
	$(VENV)\Scripts\activate && $(PYTHON) -m src.training.train_resnet

train-yolo:
	$(VENV)\Scripts\activate && $(PYTHON) -m src.training.train_yolo

tune:
	$(VENV)\Scripts\activate && $(PYTHON) -m src.tuning.hyperparameter_search

infer:
	$(VENV)\Scripts\activate && $(PYTHON) -m src.inference.predict

serve:
	$(VENV)\Scripts\activate && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rmdir /s /q $(VENV) 2>nul || echo "Virtual environment not found"
	rmdir /s /q __pycache__ 2>nul || echo "No __pycache__ found"
	rmdir /s /q .pytest_cache 2>nul || echo "No .pytest_cache found"
	del /q *.pyc 2>nul || echo "No .pyc files found"