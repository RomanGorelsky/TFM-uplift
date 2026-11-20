#!/bin/bash
#SBATCH --job-name=pilot10
#SBATCH --partition=h100
#SBATCH --qos=qos-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd ~/TFM-uplift
source slurm/env.sh

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"

python -W ignore::FutureWarning pilot/run_pilot.py \
  --sizes 2000 8000 25000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods random lgbm_t lgbm_s lgbm_t_tuned lgbm_s_tuned tabpfn_t tabpfn_s causalpfn \
  --device cuda \
  --out results/pilot10

python -W ignore::FutureWarning pilot/summarize.py --out results/pilot10
echo "finished"
