#!/bin/bash
# Reviewer check (docs/RIFT_GOTCHA_Tune.md B58, B59): does a per-pass phase correction focus the MEASURED TRAIN data
# better in 3D? Coherent backprojections (middle pulse and every pulse) per shard variant, scored against the
# data-frame Camry mesh with eval_gotcha_geometry_pvc.py, plus per-column height focus. No model, TRAIN rows only.
# Knobs (':' lists, so they travel in --export=ALL,...):
#   DATA_IMAGE_VARIANTS     images to compute (default raw:carphase). raw = uncorrected, carphase = carphase_b56,
#                           carphase_<tag> = camry_box_v2_provisional_carphase_<tag>.
#   DATA_IMAGE_SCORE        variants to score, existing npz reused (default: DATA_IMAGE_VARIANTS)
#   DATA_IMAGE_SCORES_JSON  score file name under data_image/ (default geometry_scores.json); the height-focus file
#                           takes the same name with geometry_scores -> height_focus
#SBATCH --job-name=gotcha-data-image
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" RIFT_ACCELERATOR=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONUNBUFFERED=1
BASE=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_filtered_shards_20260923
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/data_image
VARIANTS=${DATA_IMAGE_VARIANTS:-raw:carphase}
SCORE=${DATA_IMAGE_SCORE:-$VARIANTS}
SCORES=${DATA_IMAGE_SCORES_JSON:-geometry_scores.json}
FOCUS=${SCORES/geometry_scores/height_focus}
shards() {
  case $1 in
    raw) echo "$BASE/camry_box_v2_provisional/keep6/New_Transfer/shards" ;;
    carphase) echo "$BASE/camry_box_v2_provisional_carphase_b56/keep6/New_Transfer/shards" ;;
    carphase_*) echo "$BASE/camry_box_v2_provisional_$1/keep6/New_Transfer/shards" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
mkdir -p "$OUT"
hostname; date; git -C "$PWD" rev-parse HEAD
echo "variants: $VARIANTS score: $SCORE scores: $SCORES focus: $FOCUS"
rc=0
for variant in ${VARIANTS//:/ }; do
  S=$(shards "$variant") || exit 3
  [ -f "$S/pass1_hh.npz" ] || { echo "missing shards $S"; exit 3; }
  python -W ignore -u scripts_pvc/gotcha_data_coherent_image.py --shard-root "$S" --label "$variant middle pulse" --output "$OUT/${variant}_middle.npz" --threads ${SLURM_CPUS_PER_TASK:-16} || rc=1
  python -W ignore -u scripts_pvc/gotcha_data_coherent_image.py --shard-root "$S" --all-pulses --label "$variant all pulses" --output "$OUT/${variant}_all.npz" --threads ${SLURM_CPUS_PER_TASK:-16} || rc=1
done
RUNS=(); FIELDS=()
for variant in ${SCORE//:/ }; do
  for kind in middle all; do
    RUNS+=(--run "field=$OUT/${variant}_${kind}.npz:data_energy:${variant}_${kind}")
    FIELDS+=(--field "${variant}_${kind}=$OUT/${variant}_${kind}.npz:data_energy")
  done
done
rm -f "$OUT/$SCORES" "$OUT/$FOCUS"
python -W ignore -u scripts_pvc/eval_gotcha_geometry_pvc.py --mesh-dir data/meshes/camry_xv20_data_frame_box_v2 \
  "${RUNS[@]}" --thresholds 0.05 0.1 0.15 0.3 0.4 0.5 --output "$OUT/$SCORES" || rc=1
python -W ignore -u scripts_pvc/gotcha_image_height_focus.py "${FIELDS[@]}" --output "$OUT/$FOCUS" || rc=1
date; echo "EXIT=$rc"; exit $rc
