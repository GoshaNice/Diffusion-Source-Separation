from typing import List

import torch
from torch import Tensor
import numpy as np

from src.base.base_metric import BaseMetric
from src.metric.utils import calc_snr


class SNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch_prediction: Tensor, batch_target: Tensor, **kwargs):
        snrs = []
        predictions = batch_prediction.cpu().detach().numpy()
        targets = batch_target.cpu().detach().numpy()
        for prediction, target in zip(predictions, targets):
            snr = calc_snr(prediction, target)
            snrs.append(snr)
        return sum(snrs) / len(snrs)
