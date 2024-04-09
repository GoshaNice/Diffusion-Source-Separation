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

download_checkpoint:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yhJX9IyXZ1L1SbFbhW0gwpipghsJwI9w" -O pretrained_models/spexplus/model_best.pth && rm -rf /tmp/cookies.txt

download_checkpoint_spexplus:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1x_k9Iv5NHHSrjOCHkklbiRQ-VjUGI89D" -O pretrained_models/spexplus/checkpoint-epoch50_spex.pth && rm -rf /tmp/cookies.txt

test_model:
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_clean.json -b 1

train:
	poetry run python train.py -c src/ss_config.json

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