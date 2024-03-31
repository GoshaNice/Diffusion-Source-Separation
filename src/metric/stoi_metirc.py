from torch import Tensor

from src.base.base_metric import BaseMetric
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs, extended)

    def __call__(self, prediction: Tensor, target: Tensor, **kwargs):
        # prediction = prediction.squeeze(1)
        prediction, target = self.pad_to_target(prediction, target)
        stoi = self.stoi(prediction, target)

        return stoi.mean()
