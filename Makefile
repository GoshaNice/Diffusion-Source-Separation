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

experiment_main:
	poetry run python train.py -c src/configs/main.json 

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

experiment_baseline:
	poetry run python train.py -c src/configs/baseline.json

validate_baseline:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse/0418_231156/checkpoint-epoch50.pth -o baseline_output.json

validate_main:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+GlobalConditioning+FinetuneAll+SelfAttention+PostCNN/0418_232323/checkpoint-epoch50.pth -o main_output.json

validate_experiment_1:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+GlobalConditioning+FinetuneAll+NoAttention/0418_231702/checkpoint-epoch50.pth -o exp1_output.json

validate_experiment_2:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+GlobalConditioning+FinetuneAll+Attention+NoPostCNN/0418_231847/model_best.pth -o exp2_output.json

validate_experiment_3:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+NoConditioning+FinetuneAll+SelfAttention+PostCNN/0418_231852/checkpoint-epoch50.pth -o exp3_output.json

validate_experiment_4:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+LocalConditioning+FinetuneAll+SelfAttention+PostCNN/0418_232001/checkpoint-epoch50.pth -o exp4_output.json

validate_experiment_5:
	poetry run python test.py -r saved/models/TSSeparateAndDiffuse+GlobalConditioning+NoFinetune+SelfAttention+PostCNN/0418_232035/checkpoint-epoch50.pth -o exp5_output.json

validate_experiment_6:
	poetry run python test.py -r saved/models/HFG_base/0430_225752/checkpoint-epoch50.pth -o dev_exp6_output.json

validate_experiment_7:
	poetry run python test.py -r saved/models/HFG_finetune_all/0430_225849/model_best.pth -o dev_exp7_output.json

validate_spex:
	poetry run python test_spex.py -r saved/models/TSSeparateAndDiffuse+GlobalConditioning+NoFinetune+SelfAttention+PostCNN/0418_232035/checkpoint-epoch50.pth -o spex_output.json