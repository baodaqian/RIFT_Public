#!/bin/bash
#SBATCH --job-name=rift_se2_smoke_postmortem
#SBATCH --account=gts-jromberg3-ece
#SBATCH --partition=cpu-gnr
#SBATCH --qos=embers
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/Report-%j.out
#SBATCH --export=NONE

# Read-only diagnostic job: it reconstructs the retained failed smoke checkpoint
# into a new scratch directory and never resumes, trains, or edits the run.
set -euo pipefail

ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
CHECKPOINT="$ROOT/training_checkpoints/b787_sugavanam_ertin_validzero_v2p1_smoke/checkpoint_latest.pth.tar"
OUTPUT=/storage/scratch1/1/dbao31/rift_se2_v2p1_smoke_postmortem_20260821

cd "$ROOT"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing retained smoke checkpoint: $CHECKPOINT" >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to reuse diagnostic output directory: $OUTPUT" >&2
  exit 3
fi

module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
export PYTHONPATH="$ROOT:$RIFT_SITE:$EXPORT_SITE"

python -B -c 'import matplotlib, numpy, scipy, torch, skimage; from skimage.measure import marching_cubes'
python -B scripts/reconstruct_se2_smoke_geometry.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --device cpu \
  --cpu-threads "$SLURM_CPUS_PER_TASK"
