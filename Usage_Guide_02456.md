# Guide for Using MolGym-PaiNN on DTU
To recreate the data and figures used in the paper/report, one can follow the following steps. Since MolGym is a reinforcement learning model that relies on deep representation models, they take a very long time to converge upon good results. Thus the need for both running them on GPUs and on a dedicated cluster. Even on this setup there is a overall runtime of around 2400 GPU-hours for just the 6 bags with 5 seeds that we've run.  
Due to the above, this guide is aimed at running on DTU's HPC cluster.

## Setup Environment
**We shall assume that the user is able to log on to the HPC at DTU.**

> Some of the procedure described below only works for the current session, so if you are looking for a permanent setup, you can look at the section [Permanent Setup](#permanent-setup).

To ensure that all the necessary code is downloaded and compiled for running our code, the following steps need to be taken.

### Loading Modules
Log on to an interactive GPU session by running the command `voltash`. If you have other modules loaded make a call of `module purge` to ensure we start from the same clean slate.  
Get access to the DCC software stack by running 
```bash
source /dtu/sw/dcc/dcc-sw.bash
```
Load the following modules with the commands `module load <module>`:

 1. cuda/11.6
 1. python/3.9.9
 1. scipy/1.7.3
 1. matplotlib/3.4.3

### Creating a Virtual (Python) Environment
Create a directory to contain all the runs. This directory will be denoted **_pHome_**. Inside this directory run the command `python -m venv venv --prompt molgym --upgrade-deps`. This command will create a _virtual environment_ called `venv` where we can install the required packages without messing with any other python config you might have.  

> The previous command should give two errors saying that it couldn't uninstall `pip` and `setuptools`, but this is expected. It will not work if you create the virtual environment without `--upgrade-deps`, since `pip` will not become a package in the virtual environment.

Activate the virtual environment by running `source venv/bin/activate`. Create the environment variable `MOLGYM_VENV` that points to the location for the newly created virtual environment with the command
```bash
export MOLGYM_VENV=$HOME/<pHome>/venv
```
> Remember to fill in `<pHome>` with the actual path. Additionally, for a reusable setup, you can add the same line to your `.bashrc` file located in your home folder.

### Installing PyTorch
To ensure that the installed PyTorch can be run on the GPUs, we need to match it to the version of CUDA we have loaded. Thus we run the following command to ensure compatibility
```bash
python -m pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
This will take some time, but when it's done we will install the corresponding `torch-scatter` package with
```bash
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```

### Installing MolGym-PaiNN
When PyTorch has been installed the next step is to install our branch of the MolGym-PaiNN repository. To do this create a new directory in the _Virtual Environment_ directory called `molgym-painn` and in that directory run the command
```bash
git clone https://github.com/bhastrup/molgym-painn.git
```
**Then change to our branch by running `git checkout CMB`.**  
Install all the dependency packages by running
```bash
python -m pip install -r requirements.txt
```
and install MolGym-PaiNN itself by running
```bash
python -m pip install -e .
```

### Installing xTB
The last component we need is the engine for the reward function, which is the quantum chemical calculator [xTB](https://github.com/grimme-lab/xtb). The easy way to install this program is to download a pre-compiled binary, which is what we'll do here.

Create the `programs/xTB` directory and in there run the command
```bash
wget https://github.com/grimme-lab/xtb/releases/download/v6.5.1/xtb-6.5.1-linux-x86_64.tar.xz
```
Then unpack this tarball with
```bash
tar -xf xtb-6.5.1-linux-x86_64.tar.xz
```
Add the program to your `PATH` with
```bash
export PATH=$PATH:$HOME/programs/xTB/xtb-6.5.1/bin
```
You can then test you setup by moving out of the directory and running the command `xtb --version`


### Permanent Setup
If you intend to use MolGym-PaiNN several times, you can add the following lines to your `~/.bashrc` file such that you don't have to run them every time you log on. You do still need to follow the rest of [Setup Environment](#setup-environment).
```bash
source /dtu/sw/dcc/dcc-sw.bash
module load cuda/11.6
module load python/3.9.9
module load scipy/1.7.3
module load matplotlib/3.4.3

export MOLGYM_VENV=$HOME/<pHome>/venv
source $MOLGYM_VENV/bin/activate  # Optional

export PATH=$PATH:$HOME/programs/xTB/xtb-6.5.1/bin
```

## Acquiring Data
Running the model is achieved using the [`molgym-painn/scripts/run.py`](https://github.com/bhastrup/molgym-painn/blob/CMB/scripts/run.py) script. This includes an entire argument parser which makes interaction with the code very simple, as you simply start a run by calling
```bash
python $MOLGYM_VENV/molgym-painn/scripts/run.py --argument1=value1 --argument2=value2 ...
```
We have used this script as the basis for running all our data-acquisition jobs.

To aqcuire all the data seen in the paper/report, we have created a single script that starts all 120 runs. These are all submitted to the `gpuv100` GPU-queue on the HPC. 
The script is located at [`molgym-painn/scripts/molgym_runner.py`](https://github.com/bhastrup/molgym-painn/blob/CMB/scripts/molgym_runner.py), so if you've followed the setup, you would be able to start **all 120 jobs** by running the command
```bash
python $MOLGYM_VENV/molgym-painn/scripts/molgym_runner.py
```
Running 120 jobs of 6-18 hours in length is going to take quite a while, even with heavily increased priority on that queue. So as a single example, one can instead just run a single job.  
To do this, you can use the pre-generated [`example_run.sh`](https://github.com/bhastrup/molgym-painn/blob/CMB/resources/example_run.sh) file. (This file is generated directly from using `molgym_runner.py`).  
To run the example directly, make sure you are logged on to an interactive GPU-session by running `voltash`. Then create a new directory, and inside it you run the command
```bash
source $MOLGYM_VENV/molgym-painn/resources/example_run.sh
```

> Note that `example_run.sh` is set to run for 150000 steps, which took 6 hours and 37 minutes, so lower this considerably if you just want to se the code running. 5000 steps should give a feel for what's happening.

If you instead have the time to wait, you can submit it to the GPU-queue by running
```bash
bsub < $MOLGYM_VENV/molgym-painn/resources/example_run.sh
```

## Post-Processing
Whichever way you decide to perform the run, you should end up with 2 important files:
 1. `average_return.pdf`
 1. `structures_eval.xyz`

The first of these is the reward curve, which shows how well the model is performing at each evaluation. Remember here, that the reward is optimised towards the maximum. The plot itself is generated by the [`molgym-painn/scripts/plot.py`](https://github.com/bhastrup/molgym-painn/blob/CMB/scripts/plot.py) script used inside a folder generated from a run. The script is called with the command
```bash
python $MOLGYM_VENV/molgym-painn/scripts/plot.py --dir=results
```

The second of these is a "trajectory" files, which contains the molecular structure of every single evaluation. If you have a display attached to the current session, you can view the _trajectory_ by using the command
```bash
ase gui structures_eval.xyz
```
This will open a GUI, where you can flick through the various structures and rotate them by holding down the right mouse button.

The script for generating the trajectory is [`molgym-painn/scripts/structures.py`](https://github.com/bhastrup/molgym-painn/blob/CMB/scripts/structures.py). It should be run from the same directory as the `plot.py` script, and it is called as
```bash
python $MOLGYM_VENV/molgym-painn/scripts/structures.py --dir=data --symbols=<symbols>
```
where `<symbols>` is the exact same input as was used for the call to `run.py`.

## Generating Figures
The plots seen in the paper/report that depict the performance of the different agents for one specific bag have been generated by using the [`molgym-painn/scripts/comparison_plot.py`](https://github.com/bhastrup/molgym-painn/blob/CMB/scripts/comparison_plot.py) script. This script finds all runs located in subfolders of the current folder, groups them by agent and iteration, and finally it calculates mean and std. across the found seeds for every iteration. Then by running it for each of the examined bags we can get a comparison.

Running the script is done as
```bash
python $MOLGYM_VENV/molgym-painn/scripts/comparison_plot.py --name=<formula> --models <model1> <model2> ... 
```
where each `<model>` is the way that model is given in the `run.py` script. You can specify what each model is called in the legend by using `--labels <model1_label> <model2_label> ...`. More options are available and can be seen by using the `--help` flag.

## Disclaimer
_All information given in this guide was correct at the time of writing, but the passage of time might have invalidated certain parts._

_The authors take no responsibility for messed up setups caused by following the guide._
