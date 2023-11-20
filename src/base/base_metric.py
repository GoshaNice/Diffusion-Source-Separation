import torch.nn.functional as F


class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def pad_to_target(self, prediction, target):
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

    def __call__(self, **batch):
        raise NotImplementedError()
