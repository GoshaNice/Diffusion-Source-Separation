from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply
from torch import distributions


class GaussianNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.noiser = distributions.Normal(0, 0.05)

    def __call__(self, data: Tensor):
        return data + self.noiser.sample(data.size())


class RandomGaussianNoise(AugmentationBase):
    def __init__(self, p=0.5, *args, **kwargs):
        self._aug = RandomApply(GaussianNoise(*args, **kwargs), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
