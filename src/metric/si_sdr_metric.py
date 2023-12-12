from typing import List

import torch
from torch import Tensor
import numpy as np

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

    def __call__(self, prediction: List[Tensor], target: Tensor, noise: Tensor, **kwargs):
        prediction[0], target = self.pad_to_target(prediction[0], target)
        prediction[1], noise = self.pad_to_target(prediction[1], noise)
        sisdr1 = self.sisdr(prediction[0], target)
        sisdr1 += self.sisdr(prediction[1], noise)
        
        prediction[1], target = self.pad_to_target(prediction[1], target)
        prediction[0], noise = self.pad_to_target(prediction[0], noise)
        sisdr2 = self.sisdr(prediction[1], target)
        sisdr2 += self.sisdr(prediction[0], noise)
        return max(sisdr1.mean(), sisdr2.mean())
