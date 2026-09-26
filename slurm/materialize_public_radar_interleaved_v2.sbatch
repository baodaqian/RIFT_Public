#!/bin/bash
# CPU-only creation of immutable interpolation-split PublicRadar NPZs.
#SBATCH --job-name=rift_pr_split_v2
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --export=NONE

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: split materialization must run inside a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: split materialization requires Embers/cpu-small' >&2
  exit 94
}

if [[ $# -ne 1 ]]; then
  echo 'usage: materialize_public_radar_interleaved_v2.sbatch {cvdomes_camry|gotcha_pass2_hh}' >&2
  exit 2
fi

SCENE=$1
PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v1
OUTPUT_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved
SOURCE_CODE=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815

case "$SCENE" in
  cvdomes_camry|gotcha_pass2_hh) ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

SOURCE="$SOURCE_ROOT/$SCENE/$SCENE.npz"
OUTPUT="$OUTPUT_ROOT/$SCENE/$SCENE.npz"

cd "$SOURCE_CODE"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source "$SOURCE_CODE/scripts/activate_exact_numpy.sh"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
python -B "$PROJECT_ROOT/scripts/materialize_public_radar_interleaved_v2.py" \
  --scene "$SCENE" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --seed 42
