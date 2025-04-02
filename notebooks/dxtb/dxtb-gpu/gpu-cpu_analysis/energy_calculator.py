import torch

import dxtb
from dxtb._src.typing import DD


dd: DD = {"device": torch.device("cuda:0"), "dtype": torch.double}
n = 100
opts = {"scf_mode": "implicit"}
opts = {}

print(f"Running on {dd['device']} with {n} atoms.")


# numbers = torch.randint(1, 86, (n,), device=dd["device"])
choices = torch.tensor([1, 6, 7, 8, 9], device=dd["device"])
numbers = choices[torch.randint(len(choices), (n,), device=dd["device"])]

positions = torch.rand((n, 3), **dd) * 10
positions = positions.requires_grad_(True)

# warmup for CUDA
if dd["device"] is not None:
    if dd["device"].type == "cuda":
        _ = torch.rand(100, 100, **dd)
        del _

######################################################################
dxtb.timer.reset()
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd, opts=opts)
assert calc is not None


torch.cuda.synchronize()

e = calc.get_energy(positions)
torch.cuda.synchronize()

dxtb.timer.print(v=0)

######################################################################

numbers = numbers.cpu()
positions = positions.cpu()
dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

print(f"\nRunning on {dd['device']} with {n} atoms.")

dxtb.timer.reset()
calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **dd, opts=opts)
assert calc is not None

e2 = calc.get_energy(positions)

dxtb.timer.print(v=0)

print(e.sum())
print(e2.sum())