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
    --seed=42 \
    --name=C2H2O2 \
    --canvas_size=6 \
    --bag_scale=6 \
    --symbols=X,C,H,O \
    --formulas=C2H2O2 \
    --num_envs=16 \
    --vf_coef=1 \
    --entropy_coef=0.01 \
    --max_num_train_iters=5 \
    --lam=0.95 \
    --min_mean_distance=0.95 \
    --cutoff=5 \
    --discount=0.99 \
    --num_steps=150000 \
    --num_steps_per_iter=192 \
    --mini_batch_size=24 \
    --model=schnet_edge \

python $MOLGYM_VENV/molgym-painn/scripts/plot.py --dir=results
python $MOLGYM_VENV/molgym-painn/scripts/structures.py --dir=data --symbols=X,C,H,O
