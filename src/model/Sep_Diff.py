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
        self.skip_connection = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), padding="same")

    def forward(self, x):
        output = self.blocks(x)
        output = output + self.skip_connection(x)
        return output


class SeparateAndDiffuse(nn.Module):
    def __init__(
        self, 
        separator: nn.Module,
        diffwave: nn.Module,
        use_attention: bool = True,
        use_post_cnn: bool = True,
        finetune_backbone: bool = True,
        finetune_gm: bool = False,
        num_heads: int = 4
    ):
        super(SeparateAndDiffuse, self).__init__()
        self.backbone = separator
        for param in self.backbone.parameters():
            param.requires_grad = finetune_backbone

        if finetune_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()

        self.wav2spec = MelSpectrogram()
        self.GM = diffwave
        for param in self.GM.parameters():
            param.requires_grad = finetune_gm
            
        if finetune_gm:
            self.GM.train()
        else:
            self.GM.eval()

        self.ResnetHeadPhase = ResNetHead()
        self.ResnetHeadMagnitude = ResNetHead()
        
        self.use_attention = use_attention
        self.use_post_cnn = use_post_cnn
        
        if self.use_attention:
            self.self_attn_alpha = nn.MultiheadAttention(
                self.wav2spec.config.n_fft, num_heads, batch_first=True
            )
            self.self_attn_beta = nn.MultiheadAttention(
                self.wav2spec.config.n_fft, num_heads, batch_first=True
            )
        
        if self.use_post_cnn:
            self.resblock_alpha = nn.Sequential(
                nn.Conv2d(2, 32, (5, 5), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, (7, 7), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, (5, 5), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, (5, 5), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, (7, 7), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, (5, 5), padding="same"),
            )

            self.resblock_beta = nn.Sequential(
                nn.Conv2d(2, 32, (5, 5), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, (7, 7), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, (5, 5), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, (5, 5), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, (7, 7), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, (5, 5), padding="same"),
            )

    def forward(self, mix, **batch):
        """
        mix: torch.Tensor with shape (B, L) for some reasons B = 1 now
        """
        output = self.backbone(mix)  # (B, L, 2)
        output = output.squeeze().transpose(0, 1)
        # vd = output[:, :, 0]  # (B, L)
        predictions = []
        for i in range(output.shape[0]):
            vd = output[i : i + 1]  # (B, L)
            spec_vd = self.wav2spec(vd)  # (B, Mels, T)
            hop_length = self.wav2spec.config.hop_length
            vg = self.GM.infer(
                unconditional=False,
                scale=hop_length,
                condition=spec_vd.to(self.GM.device),
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

            D2 = self.ResnetHeadPhase(phase)  # (B, 2, N_fft, K)
            D1 = self.ResnetHeadMagnitude(magnitude)  # (B, 2, N_fft, K)

            Q = D1 * torch.exp(D2.mul(-1j))  # (B, 2, N_fft, K)

            alpha = Q[:, 0:1]  # (B, 1, N_fft, K)
            beta = Q[:, 1:2]  # (B, 1, N_fft, K)

            V_hat = (alpha * Vd_hat + beta * Vg_hat).squeeze(1)  # (B, N_fft, K)

            prediction_raw = torch.istft(V_hat, n_fft=self.wav2spec.config.n_fft)
            
            if not self.use_attention:
                predictions.append(prediction_raw)
                continue
            
            alpha_new = alpha.squeeze(1).transpose(1, 2).type(torch.float32)
            beta_new = beta.squeeze(1).transpose(1, 2).type(torch.float32)
            alpha_new, _ = self.self_attn_alpha(alpha_new, alpha_new, alpha_new)
            beta_new, _ = self.self_attn_beta(beta_new, beta_new, beta_new)
            
            alpha_new = alpha_new.transpose(1, 2).unsqueeze(1)
            beta_new = beta_new.transpose(1, 2).unsqueeze(1)
            
            if self.use_post_cnn:
                alpha_new = self.resblock_alpha(
                    torch.cat([alpha_new, alpha.type(torch.float32)], dim=1)
                )
                beta_new = self.resblock_beta(
                    torch.cat([beta_new, beta.type(torch.float32)], dim=1)
                )
            
            V_hat = (alpha_new * Vd_hat + beta_new * Vg_hat).squeeze(1)  # (B, N_fft, K)

            prediction_last = torch.istft(V_hat, n_fft=self.wav2spec.config.n_fft)
            
            predictions.append(prediction_last)

        return predictions
