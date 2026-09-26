#!/bin/bash
#SBATCH --job-name=rift_se2_smoke_views
#SBATCH --account=gts-jromberg3-ece
#SBATCH --partition=cpu-gnr
#SBATCH --qos=embers
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/Report-%j.out
#SBATCH --export=NONE

# Read-only visual diagnostic: rasterizes the already-audited rejected mesh
# into matched planform and nose-on views. It does not reconstruct, resume,
# train, or edit the smoke run.
set -euo pipefail

ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
POSTMORTEM=/storage/scratch1/1/dbao31/rift_se2_v2p1_smoke_postmortem_20260821
OUTPUT=/storage/scratch1/1/dbao31/rift_se2_v2p1_smoke_matched_views_20260821

cd "$ROOT"
if [[ ! -f "$POSTMORTEM/rejected_smoke_mesh.npz" || ! -f "$POSTMORTEM/rejected_smoke_mesh_audit.json" ]]; then
  echo "missing audited SE2 smoke postmortem inputs" >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to reuse matched-view diagnostic output: $OUTPUT" >&2
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

python -B -c 'import matplotlib, numpy'
python -B scripts/render_se2_smoke_topview.py \
  --mesh "$POSTMORTEM/rejected_smoke_mesh.npz" \
  --audit "$POSTMORTEM/rejected_smoke_mesh_audit.json" \
  --output-dir "$OUTPUT"
