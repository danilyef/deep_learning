import torch

# ReLU function
#https://leetgpu.com/challenges/relu

# My Answer
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    torch.where(input > 0, input, input.new_tensor(0.),  out = output)

# My Answer 2:
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output[:] = input.relu_()


# My answer 3:
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    torch.clamp(input, min=0, out=output)



import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import benchmark

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

N = 25_000_000
input = torch.randn(N, device=device)
output = torch.empty(N, device=device)

benchmark(solve, input, output, N)
