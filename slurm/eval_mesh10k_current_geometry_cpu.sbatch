#!/bin/bash
# One evaluation-only CPU array for the five Mesh10k result-slide scenes.
# Submit exactly once as --array=0-4 after checking that the v2 output root
# and an identically named live job are both absent.
#SBATCH --job-name=rift_m10k_geometry_v2
#SBATCH --account=gts-jromberg3
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=10G
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/evaluation/current_geometry_v2/slurm-%A_%a.out
#SBATCH --error=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/evaluation/current_geometry_v2/slurm-%A_%a.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: Slurm allocation required' >&2; exit 94; }
[[ "${SLURM_JOB_QOS:-}" == embers && "${SLURM_JOB_PARTITION:-}" == cpu-small ]] || {
  echo 'ERROR: evaluation requires Embers/cpu-small' >&2
  exit 94
}

PROJECT=/storage/project/r-jromberg3-0/dbao31/RIFT
DATA_ROOT=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/mesh_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k_data/objects
ASSET_ROOT=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/evaluation/source_meshes_v1
OUTPUT_ROOT=/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/evaluation/current_geometry_v2

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) SCENE=a320; LABEL=A320; NPZ=$DATA_ROOT/airliner_a320_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz; MESH=$ASSET_ROOT/A320_OpenVSP.stl ;;
  1) SCENE=x59; LABEL=X-59; NPZ=$DATA_ROOT/supersonic_x59_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz; MESH=$ASSET_ROOT/nasa_x59.stl ;;
  2) SCENE=firetruck; LABEL='fire truck'; NPZ=$DATA_ROOT/firetruck_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz; MESH=$ASSET_ROOT/kenney_firetruck.obj ;;
  3) SCENE=racecar; LABEL='race car'; NPZ=$DATA_ROOT/race_car_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz; MESH=$ASSET_ROOT/kenney_race_future.obj ;;
  4) SCENE=loader; LABEL=loader; NPZ=$DATA_ROOT/loader_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz; MESH=$ASSET_ROOT/kenney_tractor_shovel.obj ;;
  *) echo "ERROR: unknown array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 94 ;;
esac

for required in "$PROJECT/scripts/eval_mesh10k_current_geometry.py" "$NPZ" "$MESH"; do
  [[ -s "$required" ]] || { echo "ERROR: missing input $required" >&2; exit 94; }
done

OUT=$OUTPUT_ROOT/$SCENE
[[ ! -e "$OUT/current_geometry_summary.json" && ! -e "$OUT/current_fields_top_front.png" ]] || {
  echo "ERROR: refusing to overwrite an existing evaluation in $OUT" >&2
  exit 94
}
mkdir -p "$OUT"

module purge
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate RIFT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$PROJECT"

cd "$PROJECT"
python -B -u scripts/eval_mesh10k_current_geometry.py \
  --scene "$SCENE" --scene-label "$LABEL" --npz-path "$NPZ" --mesh "$MESH" --out-dir "$OUT"
