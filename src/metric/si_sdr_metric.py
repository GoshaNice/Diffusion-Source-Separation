from typing import List

import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F

from src.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


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


class SiSDRMetric(BaseMetric):
    def __init__(self, zero_mean=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        **kwargs
    ):
        self.sisdr = self.sisdr.to(prediction.device)
        prediction, target = self.pad_to_target(prediction, target)
        sisdr = self.sisdr(prediction, target)

        return sisdr.mean()
