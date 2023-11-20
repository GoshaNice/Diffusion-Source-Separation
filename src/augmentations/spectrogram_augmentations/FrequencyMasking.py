import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomFrequencyMasking(AugmentationBase):
    def __init__(self, p=0.5, *args, **kwargs):
        self._aug = RandomApply(FrequencyMasking(*args, **kwargs), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
