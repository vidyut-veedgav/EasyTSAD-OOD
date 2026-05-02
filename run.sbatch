#!/bin/bash
#SBATCH -J autoseg_full
#SBATCH -p markov_gpu
#SBATCH -C gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Always run from where you submitted the job
cd "$SLURM_SUBMIT_DIR"

# ---- Environment / Conda ----
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OODResearch

# Quick sanity prints into the log
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $SLURM_NODELIST"
echo "PWD:   $(pwd)"
python -c "import sys, torch; print('py',sys.version); print('torch',torch.__version__); print('torch cuda',torch.version.cuda); print('cuda available',torch.cuda.is_available())"

# ---- Run your pipeline ----
python Examples/run_your_algo/run_all.py