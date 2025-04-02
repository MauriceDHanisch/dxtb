import re

def parse_log_all_grid_points(log_file, keyword):
    results = []

    with open(log_file, "r") as f:
        content = f.read()
        
    # Split the content by grid computations using a separator pattern
    grid_sections = re.split(r"#{20} New Computation: n=(\d+), n_batch=(\d+) #{20}", content)
    
    # Iterate over each grid section
    for i in range(1, len(grid_sections), 3):
        n = int(grid_sections[i])
        n_batch = int(grid_sections[i + 1])
        section = grid_sections[i + 2]

        # Extract CUDA and CPU sections
        cuda_section = re.search(r"Running on cuda.*?Energy Sum:.*", section, re.DOTALL)
        cpu_section = re.search(r"Running on cpu.*?Energy Sum:.*", section, re.DOTALL)

        # Initialize results for this grid point
        grid_result = {
            "n": n,
            "n_batch": n_batch,
            "cuda": {"time": None, "percentage": None},
            "cpu": {"time": None, "percentage": None}
        }

        # Pattern to match the keyword line, capturing time and percentage
        pattern = rf"{re.escape(keyword)}\s+([\d\.]+)\s+([\d\.]+)"
        
        # Search for keyword in the CUDA section
        if cuda_section:
            match = re.search(pattern, cuda_section.group())
            if match:
                grid_result["cuda"]["time"] = float(match.group(1))
                grid_result["cuda"]["percentage"] = float(match.group(2))
        
        # Search for keyword in the CPU section
        if cpu_section:
            match = re.search(pattern, cpu_section.group())
            if match:
                grid_result["cpu"]["time"] = float(match.group(1))
                grid_result["cpu"]["percentage"] = float(match.group(2))
        
        # Append results for this grid point
        results.append(grid_result)

    return results
