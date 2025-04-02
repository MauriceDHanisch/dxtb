from itertools import product
from io import StringIO
import logging
import re
import h5py

import numpy as np
import torch
import dxtb
from dxtb._src.io import OutputHandler

# Function to strip ANSI escape sequences
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Function to load molecule data from an HDF5 file
def load_molecule_data(hdf5_file, group_name):
    with h5py.File(hdf5_file, "r") as f:
        group = f[group_name]
        atomic_numbers = torch.tensor(group["atomic_numbers"][:], dtype=torch.long)
        positions = torch.tensor(group["coordinates"][:], dtype=torch.double)
    return atomic_numbers, positions

# Define the function to run the calculation and save output to log file
def run_calculation(n, numbers, positions, device_dd, opts, log_file, num_carbons):
    # Move data to the specified device
    numbers = numbers.to(device_dd["device"])
    positions = positions.to(device_dd["device"]).requires_grad_(True)

    # Warmup for CUDA
    if device_dd["device"].type == "cuda":
        _ = torch.rand(100, 100, device=device_dd["device"], dtype=device_dd["dtype"])
        del _

    # Redirect OutputHandler console_logger to StringIO buffer
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    OutputHandler.console_logger.addHandler(handler)

    # Run calculation and capture logs
    with open(log_file, "a") as f:
        print(f"\nRunning on {device_dd['device']} with {n} atoms and {num_carbons} C atoms.", file=f)
        dxtb.timer.reset()
        
        # Run calculation
        calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **device_dd, opts=opts)
        energy = calc.get_energy(positions)
        calc.reset()
        
        dxtb.timer.print(v=0)
        
        # Write captured logs to file after stripping ANSI codes
        handler.flush()
        log_output = strip_ansi_codes(buffer.getvalue())
        f.write(log_output)
        buffer.close()
        
        print(f"Energy Sum: {energy.sum()}", file=f)

    # Remove the handler after logging
    OutputHandler.console_logger.removeHandler(handler)

# Grid computation over a range of molecules
def grid_computation(hdf5_file, log_file=None):
    opts = {"scf_mode": "implicit", "batch_mode": 0}
    
    # Open the log file in write mode to clear previous content
    with open(log_file, "w") as f:
        f.write(f"Grid Computation Log with loaded molecule data.\n")
        f.write("="*60 + "\n")

    with h5py.File(hdf5_file, "r") as f:
        molecule_groups = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))

    # Largest molecule
    print(f"Running on {hdf5_file} with {len(molecule_groups)} molecules.")
    print(f"Largest molecule: {molecule_groups[-1]}")

    for group_name in molecule_groups:
        numbers, positions = load_molecule_data(hdf5_file, group_name)
        n = len(numbers)
        num_carbons = torch.sum(numbers == 6).item()  # Count C atoms (atomic number 6)

        # Write a visual marker for each new computation
        with open(log_file, "a") as f:
            f.write(f"\n{'#'*20} New Computation: Molecule {group_name} with {n} atoms and {num_carbons} C atoms {'#'*20}\n")
        
        # Run on GPU
        dd_cuda = {"device": torch.device("cuda:0"), "dtype": torch.double}
        run_calculation(n, numbers, positions, dd_cuda, opts, log_file, num_carbons)

        # Run on CPU
        dd_cpu = {"device": torch.device("cpu"), "dtype": torch.double}
        run_calculation(n, numbers, positions, dd_cpu, opts, log_file, num_carbons)

# Define the HDF5 file and log file paths
hdf5_file = "alkanes_data_500.hdf5"
log_file = "logs/alkane_chain_E_curve.txt"

if __name__ == "__main__":
    grid_computation(hdf5_file, log_file)
