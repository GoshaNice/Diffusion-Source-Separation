import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel
from src.preprocessing.melspec import MelSpectrogram


def angle(x: torch.Tensor):
    """
    input:
        x: torch.Tensor with shape (B, N_fft, K)
    output:
        x: torch.Tensor with shape (B, N_fft, K)
    """
    return torch.atan(x.imag / x.real)


def get_magnitude(x: torch.Tensor):
    """
    input:
        x: torch.Tensor with shape (B, N_fft, K)
    output:
        x: torch.Tensor with shape (B, N_fft, K)
    """
    return torch.abs(x)


class ResNetHead(nn.Module):
    def __init__(self, hidden_channels=[32, 32, 64, 64, 64]):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(2, 32, (3, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, (3, 3), padding="same"),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class SeparateAndDiffuse(nn.Module):
    def __init__(self, separator: nn.Module, diffwave: nn.Module):
        super(SeparateAndDiffuse, self).__init__()
        self.backbone = separator
        self.wav2spec = MelSpectrogram()
        self.GM = diffwave
        self.ResnetHeadPhase = ResNetHead()
        self.ResnetHeadMagnitude = ResNetHead()

    def forward(self, mix, **batch):
        """
        mix: torch.Tensor with shape (B, L)
        """
        output = self.backbone(mix)  # (B, L, 2)
        vd = output[:, :, 0]  # (B, L)

        spec_vd = self.wav2spec(vd)  # (B, Mels, T)
        hop_length = self.wav2spec.config.hop_length
        vg = self.GM.decode_batch(mel=spec_vd, hop_len=hop_length)  # (B, L)
        vg = vg[:, : vd.shape[1]]

        Vg_hat = torch.stft(
            vg, n_fft=self.wav2spec.config.n_fft, onesided=False, return_complex=True
        )  # (B, N_fft, K)
        Vd_hat = torch.stft(
            vd, n_fft=self.wav2spec.config.n_fft, onesided=False, return_complex=True
        )  # (B, N_fft, K)

        phase = torch.cat(
            [
                angle(Vd_hat).unsqueeze(1),
                angle(Vg_hat * torch.conj(Vd_hat)).unsqueeze(1),
            ],
            dim=1,
        )  # (B, 2, N_fft, K)

        magnitude = torch.cat(
            [get_magnitude(Vd_hat).unsqueeze(1), get_magnitude(Vg_hat).unsqueeze(1)],
            dim=1,
        )  # (B, 2, N_fft, K)

        D2 = self.ResnetHeadPhase(phase)  # (B, 2, N_fft, K)
        D1 = self.ResnetHeadMagnitude(magnitude)  # (B, 2, N_fft, K)

        Q = D1 * torch.exp(D2.mul(-1j))  # (B, 2, N_fft, K)

        alpha = Q[:, 0:1]  # (B, 1, N_fft, K)
        beta = Q[:, 1:2]  # (B, 1, N_fft, K)

        V_hat = (alpha * Vd_hat + beta * Vg_hat).squeeze(1)  # (B, N_fft, K)

        prediction = torch.istft(V_hat, n_fft=self.wav2spec.config.n_fft)
        return prediction
