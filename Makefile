setup:
	python -m venv venv
	source venv/Scripts/activate

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt # Install python dependecies 
	python -m checkpoints.checkpoint_download # Install pre-trained model

test:
	pytest -v test_api.py

docker-build:
	docker build -t 100sports-app .

docker-run:
	docker run -p 2000:2000 100sports-app