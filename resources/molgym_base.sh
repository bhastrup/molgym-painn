#!/bin/sh
#BSUB -q gpuv100
#BSUB -J MolGym_run
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 24:00
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

source /dtu/sw/dcc/dcc-sw.bash
module load cuda/11.6
### nvidia-smi
module load python/3.9.9
module load scipy/1.7.3
module load matplotlib/3.4.3

source $MOLGYM_VENV/bin/activate

python3 $MOLGYM_VENV/molgym-painn/scripts/run.py \
    --device=cuda \
    --eval_freq=1 \
    --target_kl=0.03 \
    --save_rollouts=eval \