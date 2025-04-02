from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to generate alkane chains with hydrogens added
def generate_alkane_chain_data(i):
    """Generates alkane chain data with atomic numbers and positions."""
    smiles = "C" * i  # Create a string of 'C's for the desired alkane length
    alkane = Chem.MolFromSmiles(smiles)
    alkane = Chem.AddHs(alkane)  # Add hydrogens to satisfy valency
    AllChem.EmbedMolecule(alkane, useRandomCoords=True)  # Generate 3D coordinates
    AllChem.UFFOptimizeMolecule(alkane)  # Optimize 3D structure
    
    # Get atomic numbers and positions
    conf = alkane.GetConformer()
    atomic_numbers = [atom.GetAtomicNum() for atom in alkane.GetAtoms()]
    atom_positions = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                conf.GetAtomPosition(atom.GetIdx()).y,
                                conf.GetAtomPosition(atom.GetIdx()).z] for atom in alkane.GetAtoms()])
    return i, atomic_numbers, atom_positions

# Define batch size for saving
batch_size = 10
alkane_chain_data = []
saved_indices = set()

# Limit to 4 threads to avoid using all CPU cores
num_threads = 64

with ThreadPoolExecutor(max_workers=num_threads) as executor, h5py.File("alkanes_data_500.hdf5", "w") as h5file:
    futures = {executor.submit(generate_alkane_chain_data, i): i for i in range(1, 500)}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating and Saving Data"):
        try:
            result = future.result()
            alkane_chain_data.append(result)

            # Save data in batches
            if len(alkane_chain_data) >= batch_size:
                for i, atomic_numbers, atom_positions in alkane_chain_data:
                    if i not in saved_indices:
                        # Create a group for each alkane molecule
                        group_name = f"alkane_{i}_carbons"
                        molecule_group = h5file.create_group(group_name)

                        # Store metadata: number of atoms
                        molecule_group.attrs['num_atoms'] = len(atomic_numbers)
                        molecule_group.attrs['name'] = group_name

                        # Store atomic numbers and coordinates within the group
                        molecule_group.create_dataset("atomic_numbers", data=atomic_numbers)
                        molecule_group.create_dataset("coordinates", data=atom_positions)

                        # Mark as saved
                        saved_indices.add(i)

                # Clear the batch after saving
                alkane_chain_data.clear()

        except Exception as e:
            print(f"An error occurred for chain length {futures[future]}: {e}")

    # Save any remaining data that didn't reach the batch size
    for i, atomic_numbers, atom_positions in alkane_chain_data:
        if i not in saved_indices:
            group_name = f"alkane_{i}_carbons"
            molecule_group = h5file.create_group(group_name)
            molecule_group.attrs['num_atoms'] = len(atomic_numbers)
            molecule_group.attrs['name'] = group_name
            molecule_group.create_dataset("atomic_numbers", data=atomic_numbers)
            molecule_group.create_dataset("coordinates", data=atom_positions)
            saved_indices.add(i)

print("Data saved to alkanes_data_500.hdf5 with molecule groups.")
