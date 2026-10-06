#!/bin/bash
# Template SLURM job that runs one W&B sweep agent on a single GPU.
# Copy this file, fill in the #SBATCH header for your cluster, then:
#   sbatch scripts/slurm_sweep_template.sh sweeps/nc/dblp/diag_sheaf.yaml
#   SWEEP_ID=abc123 sbatch scripts/slurm_sweep_template.sh      # attach to an existing sweep
#
#SBATCH --job-name=hetsheaf-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=<your-account>
# #SBATCH --partition=<your-partition>

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# module purge && module load cuda/12.8   # adapt to your cluster

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

uv sync --frozen
exec scripts/run_sweep.sh "$@"
