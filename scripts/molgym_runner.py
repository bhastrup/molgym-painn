from pathlib import Path
from subprocess import run
import os

# Read the data from the base file
points_to_base_file = Path(__file__).parent.parent / "resources" / "molgym_base.sh"  # <-- This is bad practise, but it works.
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
single_dict = {"num_steps": "150000", "num_steps_per_iter": "192", "mini_batch_size": "24"}
large_dict = {"num_steps": "250000", "num_steps_per_iter": "256", "mini_batch_size": "32"}

# set all the molecules we want to make for benchmarks
all_seeds = [42, 1234, 888, 65536, 390625]
all_formulas = ["C2H2O2", "CH3NO", "CH4O", "C3H5NO3", "C4H7N", "C3H8O"]
all_symbols = ["X,C,H,O", "X,C,H,N,O", "X,C,H,O", "X,C,H,N,O", "X,C,H,N", "X,C,H,O"]
all_sizes = [6, 6, 6, 12, 12, 12]

# create for loops that creates the desired combinations of parameters
for model in ["painnMulti" ]:  # <-- !!! set you model name as well !!!
    single_dict["model"] = model
    large_dict["model"] = model

    for seed in all_seeds:

        for size, formula, symbols in zip(all_sizes, all_formulas, all_symbols):
            current_data = data.copy()

            # first create a new folder
            folder_name = f"run_{model}_{seed}_{formula}"
            Path(folder_name).mkdir(exist_ok=True)

            # set the wanted molecules
            molecule_dict = dict(
                seed=seed,
                name=formula,
                canvas_size=size,
                bag_scale=size,
                symbols=symbols,
                formulas=formula,
            )

            # append data to the data list
            for key, value in molecule_dict.items():
                current_data.append(f"    --{key}={value} \\\n")

            for key, value in common.items():
                current_data.append(f"    --{key}={value} \\\n")

            # Choose the right settings based on size
            settings = large_dict if size == 12 else single_dict
            for key, value in settings.items():
                current_data.append(f"    --{key}={value} \\\n")

            # Post-processing code
            current_data.append("\npython $PROJECT_VENV/molgym-painn/scripts/plot.py --dir=results\n")
            current_data.append(f"python $PROJECT_VENV/molgym-painn/scripts/structures.py --dir=data --symbols={molecule_dict['symbols']}")

            # create name for new file 
            new_file_name = f"molgym_{model}_{seed}_{formula}"

            # write the new lines to the new file
            with open(f"{folder_name}/{new_file_name}", 'w', encoding='utf-8') as file:
                file.writelines(current_data)

            # run the code 
            os.chdir(folder_name)
            run(f"bsub < {new_file_name}", shell=True)
            os.chdir("..")
