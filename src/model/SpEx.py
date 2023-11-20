import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.beta = nn.parameter.Parameter(torch.zeros(channel_size, 1))
        self.gamma = nn.parameter.Parameter(torch.ones(channel_size, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        x = self.gamma * (x - mean) / torch.sqrt(var + 1e-6) + self.beta
        return x


class TCNBlock_extractor(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        is_first=False,
        spk_embed_dim=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels + spk_embed_dim * is_first,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.PReLU(),
            GlobalLayerNorm(channel_size=out_channels),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                groups=out_channels,
                padding=1,
            ),
            nn.PReLU(),
            GlobalLayerNorm(channel_size=out_channels),
            nn.Conv1d(
                in_channels=out_channels, out_channels=in_channels, kernel_size=1
            ),
        )

    def forward(self, x, speaker_embedding=None):
        if speaker_embedding is not None:
            output = torch.cat([x, speaker_embedding], dim=1)
            output = self.block(output) + x
        else:
            output = self.block(x) + x

        return output


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.need_resample = in_channels != out_channels
        self.resample = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.block2 = nn.Sequential(nn.PReLU(), nn.MaxPool1d(kernel_size=(3)))

    def forward(self, x):
        if self.need_resample:
            output = self.block1(x) + self.resample(x)
        else:
            output = self.block1(x) + x
        output = self.block2(output)
        return output


class SpExPlus(nn.Module):
    def __init__(
        self,
        L1: int = 2,
        L2: int = 4,
        L3: int = 8,
        N: int = 16,
        proj_dim: int = 32,
        tcn_extractor_hidden: int = 32,
        num_speakers: int = 248,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_short = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=N, kernel_size=L1, stride=L1 // 2),
            nn.ReLU(),
        )
        self.encoder_middle = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=N,
                kernel_size=L2,
                stride=L1 // 2,
                padding=(L2 - L1) // 2,
            ),
            nn.ReLU(),
        )
        self.encoder_long = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=N,
                kernel_size=L3,
                stride=L1 // 2,
                padding=(L3 - L1) // 2,
            ),
            nn.ReLU(),
        )

        self.norm_speaker_extractor = nn.LayerNorm(normalized_shape=3 * N)
        self.norm_speaker_encoder = nn.LayerNorm(normalized_shape=3 * N)

        self.proj_extractor = nn.Conv1d(
            in_channels=3 * N, out_channels=proj_dim, kernel_size=1
        )

        blocks_stack1 = [
            TCNBlock_extractor(in_channels=proj_dim, out_channels=tcn_extractor_hidden)
            for _ in range(7)
        ]
        blocks_stack2 = [
            TCNBlock_extractor(in_channels=proj_dim, out_channels=tcn_extractor_hidden)
            for _ in range(7)
        ]
        blocks_stack3 = [
            TCNBlock_extractor(in_channels=proj_dim, out_channels=tcn_extractor_hidden)
            for _ in range(7)
        ]
        blocks_stack4 = [
            TCNBlock_extractor(in_channels=proj_dim, out_channels=tcn_extractor_hidden)
            for _ in range(7)
        ]

        self.tch_stack1_butt = TCNBlock_extractor(
            in_channels=proj_dim,
            out_channels=tcn_extractor_hidden,
            is_first=True,
            spk_embed_dim=proj_dim,
        )
        self.tcn_stack1_extractor = nn.Sequential(*blocks_stack1)
        self.tch_stack2_butt = TCNBlock_extractor(
            in_channels=proj_dim,
            out_channels=tcn_extractor_hidden,
            is_first=True,
            spk_embed_dim=proj_dim,
        )
        self.tcn_stack2_extractor = nn.Sequential(*blocks_stack2)
        self.tch_stack3_butt = TCNBlock_extractor(
            in_channels=proj_dim,
            out_channels=tcn_extractor_hidden,
            is_first=True,
            spk_embed_dim=proj_dim,
        )
        self.tcn_stack3_extractor = nn.Sequential(*blocks_stack3)
        self.tch_stack4_butt = TCNBlock_extractor(
            in_channels=proj_dim,
            out_channels=tcn_extractor_hidden,
            is_first=True,
            spk_embed_dim=proj_dim,
        )
        self.tcn_stack4_extractor = nn.Sequential(*blocks_stack4)

        self.speaker_embedding = nn.Sequential(
            nn.Conv1d(in_channels=3 * N, out_channels=proj_dim, kernel_size=1),
            ResNetBlock(in_channels=proj_dim, out_channels=proj_dim),
            ResNetBlock(in_channels=proj_dim, out_channels=tcn_extractor_hidden),
            ResNetBlock(
                in_channels=tcn_extractor_hidden, out_channels=tcn_extractor_hidden
            ),
            nn.Conv1d(
                in_channels=tcn_extractor_hidden,
                out_channels=tcn_extractor_hidden,
                kernel_size=1,
            ),
        )

        self.mask1 = nn.Sequential(
            nn.Conv1d(in_channels=proj_dim, out_channels=N, kernel_size=1), nn.ReLU()
        )
        self.mask2 = nn.Sequential(
            nn.Conv1d(in_channels=proj_dim, out_channels=N, kernel_size=1), nn.ReLU()
        )
        self.mask3 = nn.Sequential(
            nn.Conv1d(in_channels=proj_dim, out_channels=N, kernel_size=1), nn.ReLU()
        )

        self.decoder_short = nn.ConvTranspose1d(
            N, 1, kernel_size=L1, stride=L1 // 2, bias=True
        )
        self.decoder_middle = nn.ConvTranspose1d(
            N, 1, kernel_size=L2, stride=L1 // 2, bias=True
        )
        self.decoder_long = nn.ConvTranspose1d(
            N, 1, kernel_size=L3, stride=L1 // 2, bias=True
        )

        self.classifier = nn.Linear(proj_dim, num_speakers)

    def forward(self, mix, ref, **batch):
        """
        mix, ref - (batch_size, length)
        """
        mix = mix.unsqueeze(1)  # (batch_size, 1, Len1)
        ref = ref.unsqueeze(1)  # (batch_size, 1, Len2)

        x1 = self.encoder_short(ref)  # (batch_size, N, Len3)
        x2 = self.encoder_middle(ref)  # (batch_size, N, Len3)
        x3 = self.encoder_long(ref)  # (batch_size, N, Len3)

        x = torch.cat([x1, x2, x3], dim=1)  # (batch_size, 3N, Len3)

        y1 = self.encoder_short(mix)  # (batch_size, N, L4)
        y2 = self.encoder_middle(mix)
        y3 = self.encoder_long(mix)

        y = torch.cat([y1, y2, y3], dim=1)  # (batch_size, 3N, L4)

        x = self.norm_speaker_encoder(x.transpose(1, 2)).transpose(1, 2)
        v = self.speaker_embedding(x)

        y = self.norm_speaker_extractor(y.transpose(1, 2)).transpose(1, 2)
        y = self.proj_extractor(y)

        ref_T = (ref.shape[-1] - self.L1) // (self.L1 // 2) + 1
        ref_T = ((ref_T // 3) // 3) // 3
        v = (torch.sum(v, -1) / ref_T).view(v.shape[0], -1, 1).float()

        v = v.transpose(1, 2)
        probs = self.classifier(v)
        v = v.transpose(1, 2)
        v = v.repeat(1, 1, y.shape[-1])

        y = self.tch_stack1_butt(y, v)
        y = self.tcn_stack1_extractor(y)
        y = self.tch_stack2_butt(y, v)
        y = self.tcn_stack2_extractor(y)
        y = self.tch_stack3_butt(y, v)
        y = self.tcn_stack3_extractor(y)
        y = self.tch_stack4_butt(y, v)
        y = self.tcn_stack4_extractor(y)

        m1 = self.mask1(y)
        m2 = self.mask2(y)
        m3 = self.mask3(y)

        s1 = self.decoder_short(m1 * y1)
        s2 = self.decoder_middle(m2 * y2)
        s3 = self.decoder_long(m3 * y3)

        return s1, s2, s3, probs
