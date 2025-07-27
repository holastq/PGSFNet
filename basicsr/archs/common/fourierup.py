import os
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np

def calculate_d(x, y, M, N):
    term2 = torch.exp(1j * torch.tensor(np.pi) * x / M).cuda()
    term3 = torch.exp(1j * torch.tensor(np.pi) * y / N).cuda()
    term4 = torch.exp(1j * torch.tensor(np.pi) * (x/M + y/N)).cuda()

    result = term1 + term2 + term3 + term4
    return torch.abs(result) / 4


def get_D_map_optimized(feature):
    B, C, H, W = feature.shape
    d_map = torch.zeros((1, 1, H, W), dtype=torch.float32).cuda()
    
    #Create a grid to store the indices of all (i, j) pairs 
    i_indices = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1).repeat(1, 1, 1, W).cuda()
    j_indices = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W).repeat(1, 1, H, 1).cuda()
    
    # Compute d_map using vectorization operations
    d_map[:, :, :, :] = calculate_d(i_indices, j_indices, H, W)
    
    return d_map

class freup_AreadinterpolationV2(nn.Module):
    def __init__(self, channels, upscale):
        super(freup_AreadinterpolationV2, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels, channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels, channels,1,1,0))

        self.post = nn.Conv2d(channels, channels,1,1,0)

        self.upscale = upscale

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        
        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        
        output = torch.fft.ifft2(out)
        output = torch.abs(output)
        d_map= get_D_map_optimized(x)
        crop = torch.zeros_like(x)
        crop[:,:,:,:] = output[:,:,:H,:W]
        crop = crop / d_map 
        crop = F.interpolate(crop, (self.upscale *H, self.upscale *W))
        # print(x.shape)# 1, 3, H, W
        # print(crop.shape)# 1, 3 ,2H, 2W

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, channels, upscale):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

        self.upscale = upscale

    def forward(self, x):

        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (self.upscale, self.upscale))
        pha_fuse = torch.tile(Pha, (self.upscale, self.upscale))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)