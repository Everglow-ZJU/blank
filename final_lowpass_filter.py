# import torch
# t = torch.rand(10, 10)
# map = torch.fft.rfft2(t)
# torch.set_printoptions(threshold=float("inf"))
# print(map.real)

import torch
import torch.fft as fft

def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    
    fft_input = fft.rfft2(input)
    return fft.irfft2(fft_input * kernel, s=input.shape[-2:])
