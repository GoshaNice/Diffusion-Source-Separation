from torch import Tensor

from src.base.base_metric import BaseMetric
from wvmos import get_wvmos
import torch


class WVMOSMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_wvmos(cuda=True)
        self.model.eval()

    def __call__(self, prediction: Tensor, **kwargs):
        with torch.no_grad():
            res = self.model(prediction)
        return res.mean()
