import math
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.autograd import Variable

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    # move to same device as t_right
    pi = torch.tensor(math.pi, device=t_right.device, dtype=t_right.dtype)

    y_right= torch.sin(2*pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,torch.ones(1, device=t_right.device, dtype=t_right.dtype),y_right])

    return y

class Filterbank(nn.Module):
    def __init__(self, N_filt, filt_dim, fs):
        super(Filterbank, self).__init__()

        self.fs = fs
        self.N_filt = N_filt
        self.Filt_dim = filt_dim

        # Mel Initialization of the filterbanks -> code from SincNet (dnn_models.py)
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1)
        b2=np.roll(f_cos,-1)
        b1[0]=30
        b2[-1]=(fs/2)-100

        self.freq_scale=fs*1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

    def forward(self, x):
        device = x.device # infer device from input

        N = self.Filt_dim
        t_right=torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2), device=device)

        min_freq=50.0
        min_band=50.0

        filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)
        filters = torch.zeros((self.N_filt, self.Filt_dim), device=device)

        for i in range(0, self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)

            # compute band pass
            band_pass=(low_pass2-low_pass1)
            band_pass=band_pass/torch.max(band_pass)

            # Filter window (hamming)
            n=torch.linspace(0, N, steps=N, device=device)
            window=0.54-0.46*torch.cos(2*math.pi*n/N)
            window=Variable(window.float())

            filters[i,:]=band_pass*window
        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim), padding=125) # (80, 1, 251) -> (1, 80, 3200)
        return out
    
    def get_filters(self):
        device = self.filt_b1.device
        N = self.Filt_dim
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2), device=device)

        min_freq = 50.0
        min_band = 50.0

        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)
        filters = torch.zeros((self.N_filt, self.Filt_dim), device=device)

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)

            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / torch.max(band_pass)

            n = torch.linspace(0, N, steps=N, device=device)
            window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
            filters[i, :] = band_pass * window.float()
        
        return filters.cpu().detach().numpy()