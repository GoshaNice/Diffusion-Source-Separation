import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel
from src.preprocessing.melspec import MelSpectrogram


class SeparateAndDiffuse(nn.Module):
    def __init__(self, separator: nn.Module, diffwave: nn.Module):
        super(SeparateAndDiffuse, self).__init__()
        self.backbone = separator
        self.wav2spec = MelSpectrogram()
        self.GM = diffwave
        
    def forward(self, mix):
        vd, _ = self.backbone(mix)
        spec_vd = self.wav2spec(vd)
        vg = self.GM(spec_vd)
        
        Vg = torch.stft(vg, n_fft=1024, return_complex=False)
        Vd = torch.stft(vd, n_fft=1024, return_complex=False)
        Vd[:,:,:,1] = -Vd[:,:,:,1]
        
        Vdg_hadam = Vg * Vd
        
        Vg_re = Vg[:,:,:,0]
        Vg_im = Vg[:,:,:,1]
        
        angle_Vg = torch.atan(Vg_im / Vg_re)
        angle_Vdg = torch.atan()
        
        
        
        return s1