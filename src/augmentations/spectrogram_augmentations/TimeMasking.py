import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomTimeMasking(AugmentationBase):
    def __init__(self, p: float = 0.5, *args, **kwargs):
        self._aug = RandomApply(TimeMasking(*args, **kwargs), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
