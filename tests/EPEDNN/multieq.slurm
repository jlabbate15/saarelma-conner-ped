#!/bin/bash
#SBATCH --job-name=multieq
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --output=multieq_%j.out
#SBATCH --error=multieq_%j.err
#SBATCH --chdir=/mnt/homes_global/jal2351/software/saarelma-conner-ped/tests/EPEDNN

set -euo pipefail

# Firedrake venv (also sets PETSc env vars and MPLBACKEND=Agg)
source /mnt/codes/jal2351/venv/bin/activate

# Keep threading to a single CPU for this serial run
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python epednn_multieq_profilesloop_scloop.py

echo "Job finished at $(date)"
