import numpy as np
import torch
import dxtb
from dxtb.typing import DD
from tad_mctc.data.molecules import mols as samples
from dxtb import Calculator

# Common parameters for dxtb
sample = samples["vancoh2"]
numbers = sample["numbers"]
positions = sample["positions"]
charges = torch.tensor(0.0, dtype=torch.double)

# Batching
N_BATCH = int(8000/len(numbers))
opts = {"batch_mode": 1} # same molecule -> batch_mode = 2 because no padding needed!
numbers = torch.stack([numbers]*N_BATCH)
positions = torch.stack([positions]*N_BATCH)
charges = torch.stack([charges]*N_BATCH)

# Global options for calculator
options = dict(opts, **{"scf_mode": "implicit"})

def run_energy_calculation(device, numbers, positions, charges):
    """Run energy calculation on specified device with detailed timers."""
    dxtb.timer.reset()

    dd: DD = {"dtype": torch.double, "device": device}
    device_name = "GPU" if device.type == "cuda" else "CPU"
    print(f"\nRunning energy calculation on {device_name}...")

    # Reset and start the timer for the setup phase

    # Move numbers and positions to the selected device
    numbers = numbers.to(device)
    positions = positions.clone().to(device).requires_grad_(True)
    charges = charges.to(device)

    # Initialize the dxtb calculator
    calc = Calculator(numbers, dxtb.GFN1_XTB, opts=options, **dd)

    # Measure time for energy calculation
    if device.type == "cuda":
        torch.cuda.synchronize()
    energy = calc.get_energy(positions, charges)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Print timing information and the calculated energy
    print(dxtb.timer.print(v=1))
    print(f"Total energy on {device_name}: {energy.sum().item()}")

    return energy

# Check if GPU is available, run on GPU and then on CPU
if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
    gpu_energy = run_energy_calculation(gpu_device, numbers, positions, charges)


print(f"\nNumber of atoms: {numbers.shape[1]}")
print(f"Batch size: {N_BATCH}")
print(f"Total number of atoms: {numbers.shape[1] * N_BATCH}")   

# Run on CPU
cpu_device = torch.device("cpu")
cpu_energy = run_energy_calculation(cpu_device, numbers, positions, charges)
