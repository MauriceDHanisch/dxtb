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

# Function to create a batch of the same molecule
def create_batched_data(numbers, positions, n_batch):
    batched_numbers = torch.stack([numbers]*n_batch)
    batched_positions = torch.stack([positions]*n_batch)
    charges = torch.zeros((n_batch,), dtype=torch.double)  # Assume zero charges for all
    return batched_numbers, batched_positions, charges

# Define the function to run the calculation and save output to log file
def run_calculation(n, n_batch, numbers, positions, charges, device_dd, opts, log_file, n_carbons):
    # Move data to the specified device
    numbers = numbers.to(device_dd["device"])
    positions = positions.to(device_dd["device"]).requires_grad_(True)
    charges = charges.to(device_dd["device"])

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
        print(f"\nRunning on {device_dd['device']} with {n} atoms, {n_carbons} C atoms, and {n_batch} batches.", file=f)
        dxtb.timer.reset()
        
        # Run calculation
        calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, **device_dd, charges=charges, opts=opts)
        energy = calc.get_energy(positions, chrg=charges)
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

# Main function to iterate over HDF5 file and compute for given batch range
def grid_computation(hdf5_file, nbatch, log_file=None):
    opts = {"scf_mode": "implicit", "batch_mode": 1}
    
    # Open the log file in write mode to clear previous content
    with open(log_file, "w") as f:
        f.write(f"Grid Computation Log for molecules from {hdf5_file}.\n")
        f.write("="*60 + "\n")

    with h5py.File(hdf5_file, "r") as f:
        molecule_groups = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))

    for group_name in molecule_groups:
        # Load molecule data
        numbers, positions = load_molecule_data(hdf5_file, group_name)
        n = len(numbers)  # Total number of atoms
        n_carbons = (numbers == 6).sum().item()  # Count of carbon atoms

        # Log nat and n C for the molecule
        with open(log_file, "a") as f:
            f.write(f"\n{'#'*20} Molecule {group_name}: {n} atoms, {n_carbons} C atoms {'#'*20}\n")

        # Run calculations for the specified batch range
        n_batch_range = np.unique(np.linspace(*nbatch, dtype=int))

        for n_batch in n_batch_range:
            # Create a batch of the same molecule duplicated `n_batch` times
            batched_numbers, batched_positions, charges = create_batched_data(numbers, positions, n_batch)

            # Run on GPU
            dd_cuda = {"device": torch.device("cuda:0"), "dtype": torch.double}
            run_calculation(n, n_batch, batched_numbers, batched_positions, charges, dd_cuda, opts, log_file, n_carbons)

            # Run on CPU
            dd_cpu = {"device": torch.device("cpu"), "dtype": torch.double}
            run_calculation(n, n_batch, batched_numbers, batched_positions, charges, dd_cpu, opts, log_file, n_carbons)

# Define the HDF5 file and batch size range
hdf5_file = "alkanes_data_500.hdf5"
nbatch = [2, 128, 64]  # Range of batch sizes
log_file = f"logs/alkane_chain_E_grid_batch.txt"

if __name__ == "__main__":
    grid_computation(hdf5_file, nbatch, log_file)
