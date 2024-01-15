import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def calc_si_sdr(est: torch.Tensor, target: torch.Tensor):
    """Calculate SI-SDR metric for two given tensors"""
    assert est.shape == target.shape, "Input and Target should have the same shape"
    alpha = (target * est).sum(dim=-1) / torch.norm(target, dim=-1) ** 2
    return 20 * torch.log10(
        torch.norm(alpha.unsqueeze(1) * target, dim=-1)
        / (torch.norm(alpha.unsqueeze(1) * target - est, dim=-1) + 1e-6)
        + 1e-6
    )


class SepDiffLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def forward(self, prediction, target, noise, **batch) -> Tensor:
        prediction[0], target = self.pad_to_target(prediction[0], target)
        prediction[1], noise = self.pad_to_target(prediction[1], noise)
        loss1 = -calc_si_sdr(prediction[0], target)
        loss1 -= calc_si_sdr(prediction[1], noise)

        prediction[1], target = self.pad_to_target(prediction[1], target)
        prediction[0], noise = self.pad_to_target(prediction[0], noise)
        loss2 = -calc_si_sdr(prediction[1], target)
        loss2 -= calc_si_sdr(prediction[0], noise)

        if loss1 < loss2:
            prediction_target = prediction[0]
            prediction_noise = prediction[1]
            final_loss = loss1.mean() / 2
        else:
            prediction_target = prediction[1]
            prediction_noise = prediction[0]
            final_loss = loss2.mean() / 2

        return prediction_target, prediction_noise, final_loss
