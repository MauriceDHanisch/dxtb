from itertools import product
from io import StringIO
import logging
import re

import numpy as np
import torch
import dxtb
from dxtb._src.io import OutputHandler

# Function to strip ANSI escape sequences
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Define the random data generation function
def generate_random_data(n, dtype):
    choices = torch.tensor([1, 6, 7, 8, 9])  # CHNOF (not specifying device here)
    numbers = choices[torch.randint(len(choices), (n,))]
    # numbers = torch.randint(1, 86, (n,))
    positions = torch.rand((n, 3), dtype=dtype) * 10
    positions = positions.requires_grad_(True)
    return numbers, positions

# Define the function to run the calculation and save output to log file
def run_calculation(n, numbers, positions, device_dd, opts, log_file):
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
        print(f"\nRunning on {device_dd['device']} with {n} atoms.", file=f)
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

# Grid computation over a range of n 
def grid_computation(nat, log_file=None):
    opts = {"scf_mode": "implicit", "batch_mode": 0}
    
    # Open the log file in write mode to clear previous content
    with open(log_file, "w") as f:
        f.write(f"Grid Computation Log with linspaces: nat {nat}.\n")
        f.write("="*60 + "\n")

    n_range = np.unique(np.linspace(*nat, dtype=int))

    for n in n_range[::-1]:
        # Generate random data once for both CPU and GPU calculations
        numbers, positions = generate_random_data(n, dtype=torch.double)

        # Write a visual marker for each new computation
        with open(log_file, "a") as f:
            f.write(f"\n{'#'*20} New Computation: n={n}{'#'*20}\n")
        
        # Run on GPU
        dd_cuda = {"device": torch.device("cuda:0"), "dtype": torch.double}
        run_calculation(n, numbers, positions, dd_cuda, opts, log_file)

        # Run on CPU
        dd_cpu = {"device": torch.device("cpu"), "dtype": torch.double}
        run_calculation(n, numbers, positions, dd_cpu, opts, log_file)

# Define the ranges for n 
nat = [500, 1000, 500]
log_file = f"logs/{nat}_curve_calc_log.txt"

if __name__ == "__main__":
    grid_computation(nat, log_file)
