import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


import src.loss as module_loss
import src.metric as module_metric
import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from wvmos import get_wvmos
import pyloudnorm as pyln
import torch.nn.functional as F
import numpy as np
from speechbrain.inference.vocoders import DiffWaveVocoder, HIFIGAN
from speechbrain.inference.separation import SepformerSeparation
from src.model.spex_plus import SpExPlus

def pad_to_target(prediction, target):
        if prediction.shape[-1] > target.shape[-1]:
            target = F.pad(
                target,
                (0, int(prediction.shape[-1] - target.shape[-1])),
                "constant",
                0,
            )
        elif prediction.shape[-1] < target.shape[-1]:
            prediction = F.pad(
                prediction,
                (0, int(target.shape[-1] - prediction.shape[-1])),
                "constant",
                0,
            )
        return prediction, target


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, resume):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # setup data_loader instances
    config["data"]["test-clean"]["datasets"][0]["args"]["limit"] = 10
    config["data"]["val"]["datasets"][0]["args"]["limit"] = 10
    dataloaders = get_dataloaders(config)

    # build model architecture
    checkpoint = torch.load("pretrained_models/spexplus/checkpoint-epoch50_spex.pth", map_location=device)
    separator = SpExPlus().to(device)
    separator.load_state_dict(checkpoint["state_dict"])
    diffwave = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")
    #sepformer.device = device
    diffwave.device = device
    model = config.init_obj(config["arch"], module_model, separator, diffwave)
    #model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(resume))
    checkpoint = torch.load(resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    #model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    #model.eval()
    
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)
    #lr_scheduler=None
    config["trainer"]["epochs"] = 1
    
    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=16,
    )
    
    trainer.train()
    model.load_state_dict(state_dict)
    model.eval()
    
    config["data"]["test-clean"]["datasets"][0]["args"]["limit"] = 1000
    config["data"]["val"]["datasets"][0]["args"]["limit"] = 1000
    dataloaders = get_dataloaders(config)

    calc_sisdr = ScaleInvariantSignalDistortionRatio()
    calc_pesq = PerceptualEvaluationSpeechQuality(16000, "wb")
    calc_stoi = ShortTimeObjectiveIntelligibility(16000, False)
    calc_wvmos = get_wvmos(cuda=torch.cuda.is_available())
    calc_wvmos.eval()
    results = []
    si_sdrs = []
    pesqs = []
    stois = []
    wvmoses = []


    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            prediction_raw, prediction = model(**batch)
            batch["prediction"] = prediction

            for i in range(len(batch["prediction"])):
                prediction_fresh = batch["prediction"][i]
                target = batch["target"][i].unsqueeze(0)
                prediction, target = pad_to_target(prediction_fresh, target)
                prediction = prediction.squeeze(0).detach().cpu().numpy()
                target = target.squeeze(0).detach().cpu().numpy()

                meter = pyln.Meter(16000) # create BS.1770 meter
                loud_prediction = meter.integrated_loudness(prediction)
                loud_target = meter.integrated_loudness(target)

                prediction = pyln.normalize.loudness(prediction, loud_prediction, -20)
                target = pyln.normalize.loudness(target, loud_target, -20)

                si_sdr = calc_sisdr(torch.from_numpy(prediction), torch.from_numpy(target))
                pesq = calc_pesq(torch.from_numpy(prediction), torch.from_numpy(target))
                stoi = calc_stoi(torch.from_numpy(prediction), torch.from_numpy(target))
                with torch.no_grad():
                    wvmos = calc_wvmos(prediction_fresh.unsqueeze(0))

                si_sdrs.append(si_sdr.item())
                pesqs.append(pesq.item())
                stois.append(stoi.item())
                wvmoses.append(wvmos.item())

                results.append(
                    {
                        "SI-SDR": si_sdr.item(),
                        "PESQ": pesq.item(),
                        "STOI": stoi.item(),
                        "WVMOS": wvmos.item(),
                    }
                )

    print("Final_metrics")
    print("SI-SDR: ", sum(si_sdrs) / len(si_sdrs))
    print("PESQ: ", sum(pesqs) / len(pesqs))
    print("STOI: ", sum(stois) / len(stois))
    print("WVMOS: ", sum(wvmoses) / len(wvmoses))

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f))

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "mix_dir": str(test_data_folder / "mix"),
                            "refs_dir": str(
                                test_data_folder / "refs"
                            ),
                            "targets_dir": str(
                                test_data_folder / "targets"
                            )
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test-clean", None) is not None
    config["data"]["test-clean"]["batch_size"] = args.batch_size
    config["data"]["test-clean"]["n_jobs"] = args.jobs
    config["data"]["test-clean"]["datasets"][0]["args"]["limit"] = 1000

    main(config, args.output, args.resume)
