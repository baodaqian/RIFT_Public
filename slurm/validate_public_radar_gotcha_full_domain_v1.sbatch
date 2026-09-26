#!/bin/bash
# CPU release gate for the fresh GOTCHA full-domain trainer and real archive.
#SBATCH --job-name=rift_gotcha_full_v1_gate
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=10G
#SBATCH --time=00:30:00
#SBATCH --export=NONE

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: GOTCHA full-domain v1 validation requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: GOTCHA full-domain v1 validation requires Embers/cpu-small' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
GOTCHA_NPZ=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved/gotcha_pass2_hh/gotcha_pass2_hh.npz

for path in \
  "$PROJECT_ROOT/scripts/public_radar_gotcha_full_domain_v1_partial.py" \
  "$PROJECT_ROOT/scripts/validate_public_radar_gotcha_full_domain_v1.py" \
  "$PROJECT_ROOT/scripts/validate_public_radar_interpolation_dense_v2.py" \
  "$PROJECT_ROOT/scripts/public_radar_camry_tuning.py" \
  "$PROJECT_ROOT/rift/public_radar_tuning.py" \
  "$PROJECT_ROOT/rift/fixed_planar_scene.py" \
  "$PROJECT_ROOT/rift/serialized_range_operator.py" \
  "$PROJECT_ROOT/rift/gotcha_domain.py" \
  "$GOTCHA_NPZ"; do
  [[ -f "$path" ]] || { echo "ERROR: missing validation path: $path" >&2; exit 94; }
done

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source scripts/activate_exact_numpy.sh
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"
export RIFT_PUBLIC_SOURCE_ROOT="$SOURCE_ROOT"

python -B "$PROJECT_ROOT/scripts/validate_public_radar_interpolation_dense_v2.py" \
  --gotcha-npz "$GOTCHA_NPZ"
python -B "$PROJECT_ROOT/scripts/validate_public_radar_gotcha_full_domain_v1.py" \
  --npz-path "$GOTCHA_NPZ"
