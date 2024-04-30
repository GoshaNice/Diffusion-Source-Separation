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
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yhJX9IyXZ1L1SbFbhW0gwpipghsJwI9w" -O default_test_model/model_best.pth && rm -rf /tmp/cookies.txt

test_model:
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_clean.json -b 1

train:
	poetry run python train.py -c src/ss_config.json

train_baseline:
	poetry run python train.py -c src/configs/baseline.json

train_exp1:
	poetry run python train.py -c src/configs/exp1.json

train_exp2:
	poetry run python train.py -c src/configs/exp2.json

train_exp3:
	poetry run python train.py -c src/configs/exp3.json

train_sepformer:
	poetry run python train.py -c src/configs/sepformer.json

validate_main:
	poetry run python test.py -r saved/models/SeparateAndDiffuse/0427_184315/checkpoint-epoch50.pth -o main_output.json

validate_baseline:
	poetry run python test.py -r saved/models/SeparateAndDiffuse_Baseline/0428_102406/checkpoint-epoch50.pth -o baseline_output.json

validate_exp1:
	poetry run python test.py -r saved/models/Exp1/0428_172955/checkpoint-epoch50.pth -o exp1_output.json

validate_exp2:
	poetry run python test.py -r saved/models/Exp2/0428_172955/checkpoint-epoch50.pth -o exp2_output.json

validate_exp3:
	poetry run python test.py -r saved/models/Exp3/0428_172955/checkpoint-epoch50.pth -o exp3_output.json

validate_sepformer:
	poetry run python test_sepformer.py -r saved/models/SeparateAndDiffuse_Baseline/0428_102406/checkpoint-epoch50.pth -o sepformer_output.json

validate_sepformer_post:
	poetry run python test.py -r saved/models/Sepformer/0429_235721/checkpoint-epoch50.pth -o sepformer_post_output.json