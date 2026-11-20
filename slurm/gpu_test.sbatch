#!/bin/bash
#SBATCH --job-name=gputest
#SBATCH --partition=h100
#SBATCH --qos=qos-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd ~/TFM-uplift
source slurm/env.sh

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import os; print('TABPFN_TOKEN set:', bool(os.environ.get('TABPFN_TOKEN')))"
python pilot/run_pilot.py --sizes 2000 --seeds 0 --device cuda --out results/gputest
echo "finished"
