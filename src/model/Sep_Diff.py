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
    eps = 1e-9
    return torch.atan(x.imag / (x.real + eps))


def get_magnitude(x: torch.Tensor):
    """
    input:
        x: torch.Tensor with shape (B, N_fft, K)
    output:
        x: torch.Tensor with shape (B, N_fft, K)
    """
    return torch.abs(x)


class ResNetHead(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=[32, 32, 64, 64, 64]):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(input_channels, 32, (3, 3), padding="same"),
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

    def forward(self, x, speaker_embedding=None):
        if speaker_embedding is not None:
            x = torch.cat([x, speaker_embedding], axis=1)
        x = self.blocks(x)
        return x


class SeparateAndDiffuse(nn.Module):
    def __init__(
        self,
        separator: nn.Module,
        diffwave: nn.Module,
        local_condition=True,
        finetune_backbone=False,
        finetune_gm=False,
    ):
        super(SeparateAndDiffuse, self).__init__()
        self.backbone = separator
        for param in self.backbone.parameters():
            param.requires_grad = finetune_backbone
        self.wav2spec = MelSpectrogram()
        self.GM = diffwave
        for param in self.GM.parameters():
            param.requires_grad = finetune_gm
        self.ResnetHeadPhase = ResNetHead()
        self.ResnetHeadMagnitude = ResNetHead(
            input_channels=3 if local_condition else 2
        )
        self.local_condition = local_condition

    def forward(self, mix, ref, ref_length, **batch):
        """
        mix: torch.Tensor with shape (B, L) for some reasons B = 1 now
        """
        output = self.backbone(
            mix_audio=mix, reference_audio=ref, reference_audio_len=ref_length
        )
        speaker_embedding = self.backbone.get_speaker_embedding(
            reference_audio=ref, reference_audio_len=ref_length
        )  # (B, L, 2)

        vd = output["s1"]
        # vd should be resempled to 22.05khz TODO
        spec_vd = self.wav2spec(vd)  # (B, Mels, T)
        hop_length = self.wav2spec.config.hop_length
        vg = self.GM.decode_batch(
            mel=spec_vd,
            hop_len=hop_length,
            fast_sampling=True,
            fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
        )  # (B, L)
        vg = vg[:, : vd.shape[1]]

        Vg_hat = torch.stft(
            vg,
            n_fft=self.wav2spec.config.n_fft,
            onesided=False,
            return_complex=True,
        )  # (B, N_fft, K)

        Vd_hat = torch.stft(
            vd,
            n_fft=self.wav2spec.config.n_fft,
            onesided=False,
            return_complex=True,
        )  # (B, N_fft, K)

        phase = torch.cat(
            [
                angle(Vd_hat).unsqueeze(1),
                angle(Vg_hat * torch.conj(Vd_hat)).unsqueeze(1),
            ],
            dim=1,
        )  # (B, 2, N_fft, K)

        magnitude = torch.cat(
            [
                get_magnitude(Vd_hat).unsqueeze(1),
                get_magnitude(Vg_hat).unsqueeze(1),
            ],
            dim=1,
        )  # (B, 2, N_fft, K)

        if self.local_condition:
            speaker_embedding = (
                speaker_embedding.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, magnitude.shape[-1])
                .repeat(1, 1, 4, 1)
            )

        D2 = self.ResnetHeadPhase(phase)  # (B, 2, N_fft, K)
        if self.local_condition:
            D1 = self.ResnetHeadMagnitude(
                magnitude, speaker_embedding
            )  # (B, 2, N_fft, K)
        else:
            D1 = self.ResnetHeadMagnitude(magnitude)

        Q = D1 * torch.exp(D2.mul(-1j))  # (B, 2, N_fft, K)

        alpha = Q[:, 0:1]  # (B, 1, N_fft, K)
        beta = Q[:, 1:2]  # (B, 1, N_fft, K)

        V_hat = (alpha * Vd_hat + beta * Vg_hat).squeeze(1)  # (B, N_fft, K)

        prediction = torch.istft(V_hat, n_fft=self.wav2spec.config.n_fft)
        return prediction
