import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import numpy as np


def parse_curve_log(log_file, keyword):
    results = []

    with open(log_file, "r") as f:
        content = f.read()
    
    # Split the content by each new computation section
    computation_sections = re.split(r"#{20} New Computation: Molecule (.+?) with (\d+) atoms and (\d+) C atoms #{20}", content)

    # Iterate over each computation section
    for i in range(1, len(computation_sections), 4):
        molecule_name = computation_sections[i].strip()
        n_atoms = int(computation_sections[i + 1])
        n_carbons = int(computation_sections[i + 2])
        section_content = computation_sections[i + 3]
        
        # Find all runs (cuda and cpu) within this computation
        runs = re.split(r"Running on (.+?) with \d+ atoms and \d+ C atoms\.\s+Total Energy: (.+?) Hartree\.", section_content)

        # Initialize a list to store results for this molecule
        molecule_results = {
            "molecule": molecule_name,
            "n": n_atoms,
            "n_carbons": n_carbons,
            "computations": []
        }

        # Iterate over each run
        for j in range(1, len(runs), 3):
            device = runs[j].strip()     # cuda:0 or cpu
            timings_section = runs[j + 2]

            # Initialize result for this specific computation
            computation_result = {
                "device": device,
                "times": []  # Store all matched times with keywords
            }

            # Pattern to match both main keywords and indented sub-keywords
            pattern = rf"^(?:\s*-?\s*)?({re.escape(keyword)}.*)\s+([\d\.]+)\s+([\d\.]+)"
            
            # Search for all matches in the timings section
            matches = re.findall(pattern, timings_section, re.MULTILINE)
            for match in matches:
                # Append each matched keyword, time, and percentage
                computation_result["times"].append({
                    "objective": match[0].strip(),
                    "time": float(match[1]),
                    "percentage": float(match[2])
                })

            # Append the results for this computation to the molecule's computations list
            molecule_results["computations"].append(computation_result)

        # Append molecule results to the main results list
        results.append(molecule_results)

    return results




def parse_grid_log(log_file, keyword):
    results = []

    with open(log_file, "r") as f:
        content = f.read()
        
    # Split the content by each molecule section
    molecule_sections = re.split(r"#{20} Molecule (.+?): (\d+) atoms, (\d+) C atoms #{20}", content)

    # Iterate over each molecule section
    for i in range(1, len(molecule_sections), 4):
        molecule_name = molecule_sections[i].strip()
        n = int(molecule_sections[i + 1])      # Total atoms (nat)
        n_carbons = int(molecule_sections[i + 2])  # Number of carbon atoms
        section_content = molecule_sections[i + 3]
        
        # Find all computations for this molecule
        computation_sections = re.split(r"Running on (cuda:0|cpu)? with \d+ atoms, \d+ C atoms, and (\d+) batches\.", section_content)

        # Initialize a list to store results for this molecule
        molecule_results = {
            "molecule": molecule_name,
            "n": n,
            "n_carbons": n_carbons,
            "computations": []
        }
        
        # Iterate over each computation section
        for j in range(1, len(computation_sections), 3):
            device = computation_sections[j]     # cuda or cpu
            n_batch = int(computation_sections[j + 1])  # Batch size
            timings_section = computation_sections[j + 2]
            
            # Initialize result for this specific computation
            computation_result = {
                "n_batch": n_batch,
                "device": device,
                "times": []  # Store all matched times with keywords
            }

            # Pattern to match both main keywords and indented sub-keywords
            pattern = rf"^(?:\s*-?\s*)?({re.escape(keyword)}.*)\s+([\d\.]+)\s+([\d\.]+)"
            
            # Search for all matches in the timings section
            matches = re.findall(pattern, timings_section, re.MULTILINE)
            for match in matches:
                # Append each matched keyword, time, and percentage
                computation_result["times"].append({
                    "objective": match[0].strip(),
                    "time": float(match[1]),
                    "percentage": float(match[2])
                })

            # Append the results for this computation to the molecule's computations list
            molecule_results["computations"].append(computation_result)
        
        # Append molecule results to the main results list
        results.append(molecule_results)

    return results


def plot_curve_times(parsed_data, keyword="Total", dpi=1000):
    """
    Plots molecule size (number of atoms) vs computation time for CPU and GPU.

    Parameters:
    - parsed_data: List of dictionaries returned by parse_curve_log function.
    - keyword: Specific keyword to filter and plot the time for (e.g., "Total" or "Potential").
    """
    # Lists to store data
    molecule_sizes = []
    cpu_times = []
    gpu_times = []

    # Iterate over each molecule
    for molecule in parsed_data:
        n_atoms = molecule['n']
        molecule_sizes.append(n_atoms)

        # Initialize times
        cpu_time = None
        gpu_time = None

        # Iterate over computations (runs)
        for computation in molecule['computations']:
            device = computation['device']
            times = computation['times']

            # Find the specific time for the given keyword
            for time_entry in times:
                if keyword in time_entry['objective']:
                    time = time_entry['time']
                    if 'cpu' in device.lower():
                        cpu_time = time
                    elif 'cuda' in device.lower() or 'gpu' in device.lower():
                        gpu_time = time
                    break  # Exit loop once the keyword time is found

        # Append times (use None if not found)
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

    # Filter data for sorting (only include entries where both times are not None)
    sorted_data = sorted(
        [(size, cpu, gpu) for size, cpu, gpu in zip(molecule_sizes, cpu_times, gpu_times) if cpu is not None and gpu is not None]
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    apply_formatting(dpi=dpi, font_size=18)
    
    # Plot CPU and GPU times, handling None values by skipping them
    plt.plot(
        [size for size, time in zip(molecule_sizes, cpu_times) if time is not None],
        [time for time in cpu_times if time is not None],
        'o-', label='CPU', color='darkblue', linewidth=2, markersize=5
    )
    plt.plot(
        [size for size, time in zip(molecule_sizes, gpu_times) if time is not None],
        [time for time in gpu_times if time is not None],
        'o-', label='GPU', color='crimson', linewidth=2, markersize=5
    )

    # Check and plot the intersection point if available
    if sorted_data:
        molecule_sizes_sorted, cpu_times_sorted, gpu_times_sorted = zip(*sorted_data)

        # Calculate the intersection point based on filtered data
        intersection_point = molecule_sizes_sorted[
            np.argmin(np.abs(np.array(cpu_times_sorted) - np.array(gpu_times_sorted)))
        ]

        # Plot vertical line for the intersection point
        plt.axvline(intersection_point, color='black', linestyle='--', linewidth=2, label=f'Intersection Point: {intersection_point} atoms')
    else:
        print("No valid intersection point found due to missing data.")
        molecule_sizes_sorted, cpu_times_sorted, gpu_times_sorted = [], [], []

    plt.xlabel('Molecule Size (Number of Atoms)')
    plt.ylabel('Computation Time (s)')
    plt.title(f'Alkane Chain Size vs. Computation Time for CPU and GPU ({keyword})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_time_vs_n_atoms_at_batch_size(results, batch_size, keyword="Total", dpi=300, highlight_intervals=None):
    """
    Plots computation time vs number of atoms at a fixed batch size for both CPU and GPU.
    Optionally highlights regions between the curves and shows mean ratio of higher/lower in each region.

    Parameters:
    - results: List of dictionaries from parse_grid_log.
    - batch_size: The batch size to filter computations by.
    - keyword: Timing keyword to extract (e.g., "Total", "Forces").
    - dpi: Figure resolution.
    - highlight_intervals: List of [start_n, end_n] intervals to highlight (optional).
    """
    cpu_data, gpu_data = {}, {}

    for entry in results:
        n = entry["n"]
        for comp in entry["computations"]:
            if comp["n_batch"] != batch_size:
                continue
            time = next((t["time"] for t in comp["times"] if keyword in t["objective"]), None)
            if time is None:
                continue
            if comp["device"] == "cpu":
                cpu_data[n] = time
            elif comp["device"] == "cuda:0":
                gpu_data[n] = time

    ns = sorted(set(cpu_data.keys()) | set(gpu_data.keys()))
    cpu_times = [cpu_data.get(n, np.nan) for n in ns]
    gpu_times = [gpu_data.get(n, np.nan) for n in ns]

    plt.figure()
    apply_formatting(dpi=dpi, one_column=True, font_size=12)
    plt.plot(ns, cpu_times, "o-", label="CPU", color="navy", markersize=2)
    plt.plot(ns, gpu_times, "o-", label="GPU", color="crimson", markersize=2)

    if highlight_intervals:
        for start, end in highlight_intervals:
            mask = [(start <= n <= end) for n in ns]
            xs = [n for i, n in enumerate(ns) if mask[i]]
            c_vals = [cpu_times[i] for i in range(len(ns)) if mask[i]]
            g_vals = [gpu_times[i] for i in range(len(ns)) if mask[i]]

            if not xs:
                continue  # skip empty interval

            upper = np.maximum(c_vals, g_vals)
            lower = np.minimum(c_vals, g_vals)
            ratio = np.mean(np.array(upper) / np.array(lower))

            plt.fill_between(xs, c_vals, g_vals, color="forestgreen", alpha=0.3)
            mid_x = 0.5 * (start + end)
            max_y = max(max(c_vals), max(g_vals))
            offset = 0.5  # 5% vertical offset
            plt.text(
                mid_x,
                max_y + offset,
                f"avg ratio \n{ratio:.2f}×", # 
                ha="center",
                va="bottom",
                fontsize=10,
                color="forestgreen",
                bbox=dict(facecolor="white", alpha=1, edgecolor="k"),
                
            )

    plt.xlabel("Number of Atoms")
    plt.ylabel("Computation Time (s)")
    plt.title(f"Dxtb time vs Atom Count @ Batch Size = {batch_size} ({keyword})")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()




def plot_time_vs_batch_size_at_n_atoms(results, n_atoms, anchor_batch_size=2, keyword="Total", dpi=300):
    """
    Plots computation time vs batch size for the closest number of atoms to `n_atoms`,
    along with a linear scaling reference anchored at anchor_batch_size.
    """
    # Find closest n_atoms available
    available_n = [entry["n"] for entry in results]
    if not available_n:
        print("No atom counts found in results.")
        return

    closest_n = min(available_n, key=lambda x: abs(x - n_atoms))
    print(f"Using closest match: requested n={n_atoms}, using n={closest_n}")

    # Filter data for this closest_n
    cpu_times = {}
    gpu_times = {}

    for entry in results:
        if entry["n"] != closest_n:
            continue

        for computation in entry["computations"]:
            batch_size = computation["n_batch"]
            device = computation["device"]

            # Extract timing for the specified keyword
            time = next((t["time"] for t in computation["times"] if keyword in t["objective"]), None)
            if time is None:
                continue

            if device == "cpu":
                cpu_times[batch_size] = time
            elif device == "cuda:0":
                gpu_times[batch_size] = time

    # Sort batch sizes
    batch_sizes = sorted(set(cpu_times.keys()) & set(gpu_times.keys()))
    cpu_values = [cpu_times[b] for b in batch_sizes]
    gpu_values = [gpu_times[b] for b in batch_sizes]

    # Get anchor values for linear scaling
    anchor_cpu = cpu_times.get(anchor_batch_size)
    anchor_gpu = gpu_times.get(anchor_batch_size)

    cpu_linear = [anchor_cpu * (b / anchor_batch_size) for b in batch_sizes] if anchor_cpu else None
    gpu_linear = [anchor_gpu * (b / anchor_batch_size) for b in batch_sizes] if anchor_gpu else None

    # Plot
    plt.figure()
    apply_formatting(dpi=dpi, one_column=True, font_size=12)

    plt.plot(batch_sizes, cpu_values, "o-", label="CPU", color="navy", markersize=2)
    plt.plot(batch_sizes, gpu_values, "o-", label="GPU", color="crimson", markersize=2)

    if cpu_linear:
        plt.plot(batch_sizes, cpu_linear, "--", label=f"CPU (linear @{anchor_batch_size})", color="blue", alpha=0.5)
    if gpu_linear:
        plt.plot(batch_sizes, gpu_linear, "--", label=f"GPU (linear @{anchor_batch_size})", color="red", alpha=0.5)

    plt.xlabel("Batch Size")
    plt.ylabel("Computation Time (s)")
    plt.title(f"Time vs Batch Size (n_atoms = {closest_n})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()






def plot_grid_times(results, keyword="Total", to_plot=['cuda:0', 'cpu', 'difference'], dpi=1000):
    """
    Plots molecule size (number of atoms) and batch size vs computation time for CPU and GPU.
    
    Parameters:
    - results: List of dictionaries from parse_grid_log.
    - keyword: Specific keyword to filter and plot the time for (e.g., "Total" or "Potential").
    - to_plot: List specifying which plots to create (options: 'cuda:0', 'cpu', 'difference').
    - dpi: Resolution for the plot.
    """
    # Initialize dictionaries to store data for CUDA and CPU
    cuda_data = {}
    cpu_data = {}

    # Process each molecule's data
    for entry in results:
        n = entry['n']  # Number of atoms
        
        # Process each computation within this molecule
        for computation in entry['computations']:
            n_batch = computation['n_batch']
            device = computation['device']
            times = computation['times']
            
            # Find the specific time for the given keyword
            time = None
            for time_entry in times:
                if keyword in time_entry['objective']:
                    time = time_entry['time']
                    break  # Exit loop once the keyword time is found

            # Sort data based on device type and add to respective dictionary
            if device == 'cuda:0':
                cuda_data[(n, n_batch)] = time
            elif device == 'cpu':
                cpu_data[(n, n_batch)] = time

    # Extract unique values for atoms and batches, assuming regular grids
    n_values = sorted(set(key[0] for key in cuda_data.keys() | cpu_data.keys()))
    batch_values = sorted(set(key[1] for key in cuda_data.keys() | cpu_data.keys()))
    
    # Create 2D arrays for X, Y, and Z for both CUDA and CPU
    X, Y = np.meshgrid(n_values, batch_values)
    Z_cuda = np.array([[cuda_data.get((n, nb), np.nan) for n in n_values] for nb in batch_values])
    Z_cpu = np.array([[cpu_data.get((n, nb), np.nan) for n in n_values] for nb in batch_values])
    
    # Set up plot figure
    fig = plt.figure(figsize=(18, 6))
    apply_formatting(dpi=dpi)
    plot_index = 1

    # Determine common z-axis limits for consistent scale
    zmin, zmax = np.nanmin([Z_cuda, Z_cpu]), np.nanmax([Z_cuda, Z_cpu])
    azim, elev = -60, 30  # Customize angles to improve visibility

    # Plot CUDA if requested
    if 'cuda:0' in to_plot:
        ax1 = fig.add_subplot(1, len(to_plot), plot_index, projection='3d')
        ax1.plot_surface(X, Y, Z_cuda, cmap='viridis', edgecolor=(0, 0, 0, 0.2), vmin=zmin, vmax=zmax*0.5)
        ax1.set_title(f"Computation Time on CUDA:0 ({keyword})")
        ax1.set_xlabel("Number of Atoms (n)")
        ax1.set_ylabel("Number of Batches")
        ax1.set_zlabel("Computation Time (s)")
        ax1.invert_xaxis()
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_zlim(zmin, zmax)
        plot_index += 1

    # Plot the difference (CPU - CUDA) if requested
    if 'difference' in to_plot:
        time_diff = Z_cpu - Z_cuda  # Compute time difference
        ax3 = fig.add_subplot(1, len(to_plot), plot_index, projection='3d')

        norm = AsymmetricCenterNorm(vmin=np.min(time_diff)*0.7, vmax=np.max(time_diff)*0.3, center=0)

        ax3.plot_surface(X, Y, time_diff, cmap='coolwarm', edgecolor=(0, 0, 0, 0.2), norm=norm)
        # ax3.plot_surface(X, Y, time_diff, cmap='plasma', edgecolor='none')
        # ax3.contourf(X, Y, time_diff, zdir='z', offset=0, levels=[-10, 0], colors='green', alpha=0.2)
        ax3.set_title(f"Computation Time Difference (CPU - CUDA) ({keyword})")
        ax3.set_xlabel("Number of Atoms (n)")
        ax3.set_ylabel("Number of Batches")
        ax3.set_zlabel("Time Difference (s)")
        ax3.invert_xaxis()
        ax3.view_init(elev=elev, azim=azim)
        plot_index += 1

    # Plot CPU if requested
    if 'cpu' in to_plot:
        ax2 = fig.add_subplot(1, len(to_plot), plot_index, projection='3d')
        ax2.plot_surface(X, Y, Z_cpu, cmap='viridis', edgecolor=(0, 0, 0, 0.2), vmin=zmin, vmax=zmax*0.5)
        ax2.set_title(f"Computation Time on CPU ({keyword})")
        ax2.set_xlabel("Number of Atoms (n)")
        ax2.set_ylabel("Number of Batches")
        ax2.set_zlabel("Computation Time (s)")
        ax2.invert_xaxis()
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_zlim(zmin, zmax)

    # Adjust padding to ensure no labels are cut off
    plt.subplots_adjust(left=0.05, right=0.7, top=0.95, bottom=0.15)
    plt.show()


class AsymmetricCenterNorm(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, center=0, **kwargs):
        """
        Custom normalization with an asymmetric range around a center point.

        Parameters:
        - vmin: Minimum data value for the negative side (e.g., -5).
        - vmax: Maximum data value for the positive side (e.g., 1000).
        - center: The center of the colormap, usually 0 for symmetric maps.
        """
        super().__init__(vmin=vmin, vmax=vmax, **kwargs)
        self.center = center

    def __call__(self, value, clip=None):
        # Initialize an array to hold normalized values
        result = np.empty_like(value, dtype=np.float64)

        # Handle values greater than the center (map to red end of colormap)
        pos_mask = value >= self.center
        result[pos_mask] = 0.5 + 0.5 * (value[pos_mask] - self.center) / (self.vmax - self.center)

        # Handle values less than the center (map to blue end of colormap)
        neg_mask = value < self.center
        result[neg_mask] = 0.5 - 0.5 * (value[neg_mask] - self.center) / (self.vmin - self.center)

        # Adjust to make zero centered as white in colormap
        return np.ma.masked_invalid(result)



def apply_formatting(dpi=1000, one_column=True, font_size=10, third_height=False):
    FIGURE_WIDTH_1COL = 140/25.4  # For Thesis style
    FIGURE_WIDTH_2COL = 140/25.4 / 2  # For Thesis style
    FIGURE_HEIGHT_1COL_GR = FIGURE_WIDTH_1COL*2/(1 + np.sqrt(5))  # Golden ratio
    FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL*1.25

    if third_height == True:
        FIGURE_HEIGHT_1COL_GR = FIGURE_HEIGHT_1COL_GR * 1.5
        FIGURE_HEIGHT_2COL_GR = FIGURE_HEIGHT_2COL_GR * 1.5

    font_size = font_size if one_column else font_size * 1.2
    legend_font_size = font_size//1.5 if one_column else font_size//1.8

    figsize = (FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR) if one_column else (FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR)

    plt.rcParams.update({
        'font.family'         : 'serif',
        'font.size'           : font_size,  
        'figure.titlesize'    : 'medium',
        'figure.dpi'          : dpi,
        'figure.figsize'      : figsize,
        'axes.titlesize'      : 'medium',
        'axes.axisbelow'      : True,
        'xtick.direction'     : 'in',
        'xtick.labelsize'     : 'small',
        'ytick.direction'     : 'in',
        'ytick.labelsize'     : 'small',
        'image.interpolation' : 'none',
        'legend.fontsize'     : legend_font_size,
        'axes.labelsize'      : font_size,
        'axes.titlesize'      : font_size,
        'xtick.labelsize'     : font_size,
        'ytick.labelsize'     : font_size,
    })
