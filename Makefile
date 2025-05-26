setup:
	python -m venv venv
	source venv/Scripts/activate

install:
	pip install -r requirements.txt # Install python dependecies 
	python -m checkpoints.checkpoint_download # Install pre-trained model

test:
	pytest -v test_api.py