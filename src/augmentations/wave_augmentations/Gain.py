import torch_audiomentations
from torch import Tensor
from src.augmentations.random_apply import RandomApply
from src.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class RandomGain(AugmentationBase):
    def __init__(self, p=0.5, *args, **kwargs):
        self._aug = RandomApply(Gain(*args, **kwargs), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
