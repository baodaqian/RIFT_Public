#!/bin/bash
# Synthetic/read-only contract validation for the prepared A320 stabilized SE smoke.
#SBATCH --job-name=rift_a320_sevz_gate
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --tmp=2G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=/storage/project/r-jromberg3-0/dbao31/RIFT/mgr_slurm_logs/Report-%j.out

set -euo pipefail

[[ $# -eq 0 ]] || { echo 'ERROR: A320 stabilized contract validation takes no arguments' >&2; exit 94; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: Slurm allocation required' >&2; exit 94; }
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: A320 stabilized contract validation requires Embers/cpu-small' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
for required in \
  "$PROJECT/rift/sugavanam_ertin_a320_stabilized.py" \
  "$PROJECT/train_sugavanam_ertin_smoke.py" \
  "$PROJECT/rift/sugavanam_ertin_stabilized_smoke_runtime.py" \
  "$PROJECT/scripts/validate_mesh10k_a320_sugavanam_ertin_stabilized.py" \
  "$PROJECT/slurm/train_mesh10k_a320_sugavanam_ertin_stabilized_smoke_v1.sbatch" \
  /storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/a320/se/scatter/checkpoint_final.pth.tar; do
  [[ -s "$required" ]] || { echo "ERROR: missing A320 contract input: $required" >&2; exit 94; }
done

cd "$PROJECT"
module purge
module load anaconda3/2023.03
CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages
export PYTHONPATH="$PROJECT:$RIFT_SITE:$EXPORT_SITE"

python -B -m py_compile \
  rift/sugavanam_ertin_a320_stabilized.py \
  train_sugavanam_ertin_smoke.py \
  rift/sugavanam_ertin_stabilized_smoke_runtime.py \
  scripts/validate_mesh10k_a320_sugavanam_ertin_stabilized.py
python -B -u scripts/validate_mesh10k_a320_sugavanam_ertin_stabilized.py --require-stage1
