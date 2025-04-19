import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Parameters
RUNS = 2

# --- Benchmark Grid Setup ---
batch_sizes = np.linspace(2, 256, 20, dtype=int)[::-1]
matrix_sizes = np.linspace(10, 1000, 30, dtype=int)[::-1]

def symmetrize(x):
    if isinstance(x, np.ndarray):
        return (x + x.transpose(0, 2, 1)) / 2
    elif isinstance(x, torch.Tensor):
        return (x + x.transpose(-1, -2)) / 2
    else:
        return (x + jnp.transpose(x, axes=(0, 2, 1))) / 2

# --- NumPy benchmark ---
def benchmark_numpy(x, runs=RUNS):
    x = symmetrize(x)
    _ = np.linalg.eigh(x)   # warm-up
    start = time.time()
    for _ in range(runs):
        _ = np.linalg.eigh(x)
    return (time.time() - start) / runs

# --- PyTorch benchmark ---
def benchmark_torch(x_np, device, runs=RUNS):
    x = symmetrize(torch.tensor(x_np, device=device))
    _ = torch.linalg.eigh(x)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        _ = torch.linalg.eigh(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / runs

# --- JAX benchmark ---
@jax.jit
def batched_eigh_jax(x):
    return jax.vmap(jnp.linalg.eigh)(x)

def benchmark_jax(x_np, device, runs=RUNS):
    x = symmetrize(jax.device_put(x_np, device=device))
    _ = batched_eigh_jax(x)  # warm-up
    jax.block_until_ready(_)
    start = time.time()
    for _ in range(runs):
        result = batched_eigh_jax(x)
        jax.block_until_ready(result)
    return (time.time() - start) / runs

import pickle
from tqdm import tqdm

results = {
    "numpy_cpu": np.zeros((len(batch_sizes), len(matrix_sizes))),
    "torch_cpu": np.zeros((len(batch_sizes), len(matrix_sizes))),
    "torch_gpu": np.zeros((len(batch_sizes), len(matrix_sizes))),
    "jax_cpu": np.zeros((len(batch_sizes), len(matrix_sizes))),
    "jax_gpu": np.zeros((len(batch_sizes), len(matrix_sizes))),
}

# --- Run Benchmarks ---
for i, B in tqdm(enumerate(batch_sizes), total=len(batch_sizes)):
    for j, N in enumerate(matrix_sizes):
        np.random.seed(0)
        data_np = np.random.randn(B, N, N).astype(np.float32)
        data_np = 0.5 * (data_np + np.transpose(data_np, (0, 2, 1)))

        results["numpy_cpu"][i, j] = benchmark_numpy(data_np)
        results["torch_cpu"][i, j] = benchmark_torch(data_np, device="cpu")
        results["jax_cpu"][i, j] = benchmark_jax(data_np, device=jax.devices("cpu")[0])

        if torch.cuda.is_available():
            results["torch_gpu"][i, j] = benchmark_torch(data_np, device="cuda")
        else:
            results["torch_gpu"][i, j] = np.nan

        if any(d.platform == "gpu" for d in jax.devices()):
            results["jax_gpu"][i, j] = benchmark_jax(data_np, device=jax.devices("gpu")[0])
        else:
            results["jax_gpu"][i, j] = np.nan

# --- Save timings ---
with open("eigh_benchmark_results_full.pkl", "wb") as f:
    pickle.dump({"batch_sizes": batch_sizes, "matrix_sizes": matrix_sizes, "results": results}, f)