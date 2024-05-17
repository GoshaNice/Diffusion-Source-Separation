# Diffusion Source Separation

#### Implemented by: Pistsov Georgiy 202

You can find experiments details [here](https://wandb.ai/goshanice/diffusion_ss_project/overview?workspace=user-goshanice)

## Installation guide

Current repository is for Linux

(optional, not recommended) if you are trying to install it on macos run following before install:
```shell
make switch_to_macos
```

Then you run:

```shell
make install
```


## Train model:

```shell
make train
```
Config for training you can find in src/config.json


## Run any other python script:

If you want to run any other custom python script, you can just start it with "poetry run"
For example:

Instead of:

```shell
python test.py -r default_test_model/model_best.pth
```

You can use:

```shell
poetry run python test.py -r default_test_model/model_best.pth
```


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.