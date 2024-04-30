from torch import Tensor

from src.base.base_metric import BaseMetric
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(
        self,
        prediction_target: Tensor,
        prediction_noise: Tensor,
        target: Tensor,
        noise: Tensor,
        **kwargs
    ):
        # prediction = prediction.squeeze(1)
        try:
            prediction_target, target = self.pad_to_target(prediction_target, target)
            pesq = self.pesq(prediction_target, target)

            prediction_noise, target = self.pad_to_target(prediction_noise, noise)
            pesq += self.pesq(prediction_noise, noise)
            return pesq.mean() / 2
        except:
            return 1
