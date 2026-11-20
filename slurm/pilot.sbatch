#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --partition=h100
#SBATCH --qos=qos-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd ~/TFM-uplift
source slurm/env.sh

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"

python pilot/run_pilot.py \
  --sizes 2000 8000 25000 \
  --seeds 0 1 2 \
  --device cuda \
  --out results/pilot

python pilot/summarize.py --out results/pilot
echo "finished"
