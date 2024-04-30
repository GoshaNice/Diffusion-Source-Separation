import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel


class Sepformer(nn.Module):
    def __init__(
        self, 
        separator: nn.Module,
        diffwave: nn.Module,
    ):
        super(Sepformer, self).__init__()
        self.backbone = separator
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone.train()

        self.GM = diffwave
        for param in self.GM.parameters():
            param.requires_grad = False
        
    def forward(self, mix, **batch):
        """
        mix: torch.Tensor with shape (B, L) for some reasons B = 1 now
        """
        # mix should be resempled to 8khz TODO
        output = self.backbone(mix)  # (B, L, 2)
        output = output.squeeze().transpose(0, 1)
        # vd = output[:, :, 0]  # (B, L)
        predictions = []
        for i in range(output.shape[0]):
            vd = output[i : i + 1]  # (B, L)
            predictions.append(vd)

        return predictions
