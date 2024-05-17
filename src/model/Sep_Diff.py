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
        self.blocks_seq = nn.Sequential(
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

        self.blocks_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_channels, 32, (3, 3), padding="same"),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(33, 32, (3, 3), padding="same"),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(33, 64, (3, 3), padding="same"),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(65, 64, (3, 3), padding="same"),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, (3, 3), padding="same"),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.Conv2d(65, 2, (3, 3), padding="same"),
            ]
        )

    def forward(self, x, conditioning="no", speaker_embedding=None):
        if conditioning == "no":
            return self.blocks_seq(x)
        elif conditioning == "global":
            for block in self.blocks_list:
                x = torch.cat([x, speaker_embedding], axis=1)
                x = block(x)
            return x
        elif conditioning == "local":
            x = torch.cat([x, speaker_embedding], axis=1)
            return self.blocks_seq(x)


class SeparateAndDiffuse(nn.Module):
    def __init__(
        self,
        separator: nn.Module,
        diffwave: nn.Module,
        conditioning: str = "no",
        finetune_backbone: bool = False,
        finetune_gm: bool = False,
        num_heads: int = 4,
        use_attention: bool = False,
        use_post_cnn: bool = False,
    ):
        super(SeparateAndDiffuse, self).__init__()
        self.backbone = separator
        for param in self.backbone.parameters():
            param.requires_grad = finetune_backbone

        if not finetune_backbone:
            self.backbone.eval()
        else:
            self.backbone.train()

        self.wav2spec = MelSpectrogram()
        self.GM = diffwave
        for param in self.GM.parameters():
            param.requires_grad = finetune_gm
            
        if not finetune_gm:
            self.GM.eval()
        else:
            self.GM.train()

        assert conditioning in [
            "no",
            "local",
            "global",
        ], "Only no, local and global conditionings are supported"
        self.conditioning = conditioning

        self.ResnetHeadPhase = ResNetHead()
        self.ResnetHeadMagnitude = ResNetHead(
            input_channels=3 if self.conditioning != "no" else 2
        )

        self.use_attention = use_attention
        if self.use_attention:
            self.self_attn_alpha = nn.MultiheadAttention(
                self.wav2spec.config.n_fft, num_heads, batch_first=True
            )
            self.self_attn_beta = nn.MultiheadAttention(
                self.wav2spec.config.n_fft, num_heads, batch_first=True
            )

        self.use_post_cnn = use_post_cnn
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
        spec_vd = self.wav2spec(vd)  # (B, Mels, T)
        hop_length = self.wav2spec.config.hop_length
        vg = self.GM.decode_batch(
            spectrogram=spec_vd,
            hop_len=hop_length,
            #fast_sampling=True,
            #fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
        ).squeeze(1)  # (B, L)
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

        if self.conditioning != "no":
            speaker_embedding = (
                speaker_embedding.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, magnitude.shape[-1])
                .repeat(1, 1, 4, 1)
            )

        D2 = self.ResnetHeadPhase(phase)  # (B, 2, N_fft, K)
        if self.conditioning != "no":
            D1 = self.ResnetHeadMagnitude(
                magnitude,
                conditioning=self.conditioning,
                speaker_embedding=speaker_embedding,
            )  # (B, 2, N_fft, K)
        else:
            D1 = self.ResnetHeadMagnitude(magnitude)

        Q = D1 * torch.exp(D2.mul(-1j))  # (B, 2, N_fft, K)

        alpha = Q[:, 0:1]  # (B, 1, N_fft, K)
        beta = Q[:, 1:2]  # (B, 1, N_fft, K)

        V_hat = (alpha * Vd_hat + beta * Vg_hat).squeeze(1)  # (B, N_fft, K)
        prediction_raw = torch.istft(V_hat, n_fft=self.wav2spec.config.n_fft)

        if not self.use_attention:
            return prediction_raw, prediction_raw

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

        return prediction_raw, prediction_last
