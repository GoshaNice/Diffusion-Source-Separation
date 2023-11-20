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


class SpExLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

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

    def forward(self, s1, s2, s3, target, logits, target_id=None, **batch) -> Tensor:
        target = target.to(s1.device)
        s1 = s1.squeeze(1)
        s2 = s2.squeeze(1)
        s3 = s3.squeeze(1)
        logits = logits.squeeze(1)
        s1, target_s1 = self.pad_to_target(s1, target)
        s2, target_s2 = self.pad_to_target(s2, target)
        s3, target_s3 = self.pad_to_target(s3, target)

        loss = torch.zeros((s1.shape[0]), device=s1.device)
        loss -= (1 - self.alpha - self.beta) * calc_si_sdr(s1, target_s1)
        loss -= self.alpha * calc_si_sdr(s2, target_s2)
        loss -= self.beta * calc_si_sdr(s3, target_s3)
        if target_id is not None:
            self.ce = self.ce.to(logits.device)
            loss += self.gamma * self.ce(logits, target_id)
        return loss.mean()
