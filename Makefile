CODE = src

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

lint:
	poetry run pflake8 $(CODE)

format:
	#format code
	poetry run black $(CODE)

download_checkpoint_spexplus:
	mkdir pretrained_models/spextest
	gdown https://drive.google.com/uc\?id\=10MPcYDL8csaZX4R87WxGsDYV7zQyjOfU -O pretrained_models/spextest/checkpoint-epoch50_spex.pth

test_model:
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_clean.json -b 1

train:
	poetry run python train.py -c src/ss_config.json

make setup:
	python3.10 -m pip install poetry
	python3.10 -m pip install gdown
	poetry install
	mkdir pretrained_models pretrained_models/spexplus
	gdown https://drive.google.com/uc?id=10MPcYDL8csaZX4R87WxGsDYV7zQyjOfU -O pretrained_models/spexplus/checkpoint-epoch50_spex.pth
	export WANDB_API_KEY=d7a59d1f2d033191490803ece03644b895ff4bd2

experiment_1:
	poetry run python train.py -c src/configs/exp1.json

experiment_2:
	poetry run python train.py -c src/configs/exp2.json

experiment_3:
	poetry run python train.py -c src/configs/exp3.json

experiment_4:
	poetry run python train.py -c src/configs/exp4.json

experiment_5:
	poetry run python train.py -c src/configs/exp5.json

experiment_6:
	poetry run python train.py -c src/configs/exp6.json

experiment_7:
	poetry run python train.py -c src/configs/exp7.json