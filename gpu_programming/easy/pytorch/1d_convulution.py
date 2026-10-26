import torch
import torch.nn.functional as F

# 1D Convolution
# https://leetgpu.com/challenges/1d-convolution

# My answer
def conv1d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
):

    return torch.matmul(input.unfold(0, kernel.shape[0], 1), kernel, out=output)


# Efficient answer
def conv1d_fft(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
):
    
    out_size = input.shape[0] - kernel.shape[0] + 1
    
    # FFT length: enough to compute the full linear convolution
    fft_size = 1 << (input.shape[0] + kernel.shape[0] - 2).bit_length()

    # Cross-correlation = convolution with the reversed kernel
    x_fft = torch.fft.rfft(input, n=fft_size)
    k_fft = torch.fft.rfft(kernel.flip(0), n=fft_size)

    full = torch.fft.irfft(x_fft * k_fft, n=fft_size)

    # Valid convolution corresponds to indices k-1 through n-1
    output[:] = full[kernel.shape[0] - 1 : kernel.shape[0] - 1 + out_size]
    return output



import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import benchmark

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

input_size, kernel_size = 1_500_000, 2_047
input = torch.rand(input_size, device=device)
kernel = torch.rand(kernel_size, device=device)
output = torch.empty(input_size - kernel_size + 1, device=device)

benchmark(conv1d_fft, input, kernel, output)
