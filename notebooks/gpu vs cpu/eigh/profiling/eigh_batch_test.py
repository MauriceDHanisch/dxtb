import torch

B, N = 8, 300
x = torch.randn(B, N, N, device="cuda")
x = 0.5 * (x + x.transpose(-1, -2))  # Symmetrize
torch.linalg.eigh(x)
