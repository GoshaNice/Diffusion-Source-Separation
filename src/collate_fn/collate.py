import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    result_batch["mix_length"] = torch.tensor(
        [item["mix"].shape[-1] for item in dataset_items]
    )
    max_mix_length = torch.max(result_batch["mix_length"]).item()
    mixes = []
    for item in dataset_items:
        mix = item["mix"]
        mixes.append(
            F.pad(
                mix,
                (0, int(max_mix_length - mix.shape[-1])),
                "constant",
                0,
            )
        )
    result_batch["mix"] = torch.cat(mixes, dim=0)

    result_batch["ref_length"] = torch.tensor(
        [item["ref"].shape[-1] for item in dataset_items]
    )
    max_ref_length = torch.max(result_batch["ref_length"]).item()
    refs = []
    for item in dataset_items:
        ref = item["ref"]
        refs.append(
            F.pad(
                ref,
                (0, int(max_ref_length - ref.shape[-1])),
                "constant",
                0,
            )
        )
    result_batch["ref"] = torch.cat(refs, dim=0)

    result_batch["target_length"] = torch.tensor(
        [item["target"].shape[-1] for item in dataset_items]
    )
    max_target_length = torch.max(result_batch["target_length"]).item()
    targets = []
    for item in dataset_items:
        target = item["target"]
        targets.append(
            F.pad(
                target,
                (0, int(max_target_length - target.shape[-1])),
                "constant",
                0,
            )
        )
    result_batch["target"] = torch.cat(targets, dim=0)
    result_batch["target_id"] = torch.tensor(
        [item["target_id"] for item in dataset_items]
    )

    result_batch["noise_length"] = torch.tensor(
        [item["noise"].shape[-1] for item in dataset_items]
    )
    max_noise_length = torch.max(result_batch["noise_length"]).item()
    noises = []
    for item in dataset_items:
        noise = item["noise"]
        noises.append(
            F.pad(
                noise,
                (0, int(max_noise_length - noise.shape[-1])),
                "constant",
                0,
            )
        )
    result_batch["noise"] = torch.cat(noises, dim=0)

    return result_batch
