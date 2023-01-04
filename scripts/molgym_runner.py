from pathlib import Path
from subprocess import run
import os
from ase import Atoms

# Read the data from the base file
points_to_base_file = (
    Path(__file__).parent.parent /
    "resources" /
    "molgym_base.sh"
)  # Pointing to a file like this is bad practise, but it works.

with open(points_to_base_file) as file:
    data = file.readlines()

# Set the run settings
common = {
    "num_envs": "16",
    "vf_coef": "1",
    "entropy_coef": "0.01",
    "max_num_train_iters": "5",
    "lam": "0.95",
    "min_mean_distance": "0.95",
    "cutoff": "5",
    "discount": "0.99"
}
single_dict = {
    "num_steps": "150000",
    "num_steps_per_iter": "192",
    "mini_batch_size": "24"
}
large_dict = {
    "num_steps": "250000",
    "num_steps_per_iter": "256",
    "mini_batch_size": "32"
}

# Set RNG seeds and choose the bags to run
seeds =    [42, 888, 1234, 65536, 390625]
formulas = ["C2H2O2", "CH3NO", "CH4O", "C3H5NO3", "C4H7N", "C3H8O",]

# Transform formulas to required input
bags =     [Atoms(formula) for formula in formulas]
symbols =  ["X," + ','.join(bag.symbols.species()) for bag in bags]
sizes =    [len(bag) for bag in bags]

# Loop over all the permutations
for model in ["schnet_edge", "painn", "painn_equivariant", "painnMulti"]:
    single_dict["model"] = model
    large_dict["model"] = model

    for seed in seeds:

        for size, formula, symbols in zip(sizes, formulas, symbols):
            current_data = data.copy()

            # Create a run folder
            folder_name = f"run_{model}_{seed}_{formula}"
            Path(folder_name).mkdir(exist_ok=True)

            # Set the wanted molecules
            molecule_dict = dict(
                seed=seed,
                name=formula,
                canvas_size=size,
                bag_scale=size,
                symbols=symbols,
                formulas=formula,
            )

            # Append data to the data list
            for key, value in molecule_dict.items():
                current_data.append(f"    --{key}={value} \\\n")

            for key, value in common.items():
                current_data.append(f"    --{key}={value} \\\n")

            # Choose the right settings based on size
            settings = large_dict if size == 12 else single_dict
            for key, value in settings.items():
                current_data.append(f"    --{key}={value} \\\n")

            # Post-processing code
            current_data.append(
                "\npython $MOLGYM_VENV/molgym-painn/scripts/plot.py "
                "--dir=results\n"
            )
            current_data.append(
                "python $MOLGYM_VENV/molgym-painn/scripts/structures.py "
                f"--dir=data --symbols={molecule_dict['symbols']}"
            )

            # Create name for submit file 
            new_file_name = f"molgym_{model}_{seed}_{formula}"

            # Write the new lines to the new file
            with open(f"{folder_name}/{new_file_name}", 'w') as file:
                file.writelines(current_data)

            # Change into the folder and submit the run
            os.chdir(folder_name)
            run(f"bsub < {new_file_name}", shell=True)
            os.chdir("..")
