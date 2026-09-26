#!/bin/bash
# CPU contract gate for the shared PublicRadar directional-SH cache.
#SBATCH --job-name=rift_pr_shcache_gate_v1
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --tmp=4G
#SBATCH --time=00:10:00
#SBATCH --export=NONE

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo 'ERROR: SH cache validation requires a Slurm allocation' >&2
  exit 94
}
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: SH cache validation requires Embers/cpu-small' >&2
  exit 94
}

PROJECT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
SOURCE_ROOT=/storage/scratch1/1/dbao31/rift_public_radar_impl_20260815
DATA_ROOT=/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814/converted_v2_interleaved

cd "$SOURCE_ROOT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
source "$SOURCE_ROOT/scripts/activate_exact_numpy.sh"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT"

python -B "$PROJECT_ROOT/scripts/validate_public_radar_sh_basis_cache.py" \
  --real-cache cvdomes_camry \
    "$DATA_ROOT/cvdomes_camry/cvdomes_camry.npz" \
    "$DATA_ROOT/cvdomes_camry/sh_basis_degree6_v1" \
  --real-cache gotcha_pass2_hh \
    "$DATA_ROOT/gotcha_pass2_hh/gotcha_pass2_hh.npz" \
    "$DATA_ROOT/gotcha_pass2_hh/sh_basis_degree6_v1"
