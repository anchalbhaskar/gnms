#!/bin/bash
#SBATCH -J local_basic_test_overflow
#SBATCH -o /Users/anchalbhaskar/Desktop/CBU Work/gnms/logs/sweep/local_basic_test/eta_gamma_batched/%x_%A_%a.out
#SBATCH -e /Users/anchalbhaskar/Desktop/CBU Work/gnms/logs/sweep/local_basic_test/eta_gamma_batched/%x_%A_%a.err
#SBATCH -p Main
#SBATCH -q lopri
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=node-k02,node-j21,node-k05,node-k06
#SBATCH --array=100-5624

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ngym39

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

if [ -f "/imaging/astle/fp02/RNN/run.py" ]; then cd "/imaging/astle/fp02/RNN";
elif [ -d "multitask-curiosity" ]; then cd multitask-curiosity; fi

CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /Users/anchalbhaskar/Desktop/CBU Work/gnms/logs/sweep/local_basic_test/eta_gamma_batched/config_list.txt)
python -u run.py $CMD
