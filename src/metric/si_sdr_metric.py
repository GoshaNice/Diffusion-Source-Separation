from typing import List

import torch
from torch import Tensor
import numpy as np

from src.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SiSDRMetric(BaseMetric):
    def __init__(self, zero_mean=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean)

    def __call__(self, prediction: Tensor, target: Tensor, **kwargs):
        prediction = prediction.squeeze(1)
        self.sisdr = self.sisdr.to(prediction.device)
        target = target.to(prediction.device)

        prediction, target = self.pad_to_target(prediction, target)
        sisdr = self.sisdr(prediction, target)
        return sisdr.mean()
