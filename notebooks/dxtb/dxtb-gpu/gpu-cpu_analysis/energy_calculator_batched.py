import torch

import dxtb
from dxtb._src.typing import DD


dd: DD = {"device": torch.device("cuda:0"), "dtype": torch.double}
n = 100
n_batch = 30
opts = {"scf_mode": "implicit", "batch_mode": 1}

# Generate random numbers tensor of shape (n_batch, n) 
# numbers = torch.randint(1, 86, (n_batch, n), device=dd["device"])
choices = torch.tensor([1, 6, 7, 8, 9], device=dd["device"]) # CHNOF
numbers = choices[torch.randint(len(choices), (n_batch, n), device=dd["device"])]
charges = torch.zeros((n_batch,), device=dd["device"], dtype=torch.double)

# Generate random positions tensor of shape (n_batch, n, 3)
positions = torch.rand((n_batch, n, 3), **dd) * 10
positions = positions.requires_grad_(True)

# warmup for CUDA
if dd["device"] is not None:
    if dd["device"].type == "cuda":
        _ = torch.rand(100, 100, **dd)
        del _

######################################################################
print(f"\nRunning on {dd['device']} with {n} atoms and {n_batch} batches.")
dxtb.timer.reset()
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd, charges=charges, opts=opts)
e = calc.get_energy(positions, chrg=charges)

dxtb.timer.print(v=0)

######################################################################

numbers = numbers.cpu()
positions = positions.cpu()
charges = charges.cpu()
dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

print(f"\nRunning on {dd['device']} with {n} atoms and {n_batch} batches.")

dxtb.timer.reset()
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd, charges=charges, opts=opts)
e2 = calc.get_energy(positions, chrg=charges)

dxtb.timer.print(v=0)

print(e.sum())
print(e2.sum())