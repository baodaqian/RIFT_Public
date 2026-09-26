#!/bin/bash
# Rebuild the RIFT Camry viewer (A64) from every saved level-end snapshot and final checkpoint, plus the current
# checkpoint_latest of the running full-data, gain-twin and extended arms. The export runs as one CPU job
# (sbatch --wait); the page is written to $1 (default: the viewer's scratch copy). Read-only on every run directory.
set -eo pipefail
OUT_HTML=${1:?output html path}
cd /home/u.db364833/RIFT
D=/scratch/group/p.cis261724.000/RIFT_pvc_runs/rift_camry_densify
S=$D/snapshots
DI=/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/data_image
run_file() { ls $D/$1/camry_box_v2/*/frequency_stride2_all_roles_v2/rift_full_native/$2 2>/dev/null | head -1; }
RUNS=()
add() { [ -n "$2" ] && [ -f "$2" ] && RUNS+=("$1=$2"); return 0; }
add T5b_ep10 $S/T5b_ep10_pre.pt; add T5b_ep16 $S/T5b_ep16_pre.pt; add T5b_ep24 "$(run_file T5b_trilinear_lr6e-4 checkpoint_final.pt)"
for arm in M5b CM5b; do add ${arm}_ep10 $S/${arm}_ep10_pre.pt; add ${arm}_ep16 $S/${arm}_ep16_pre.pt; done
add M5b_ep24 "$(run_file M5b_trilinear_lr6e-4_1024k checkpoint_final.pt)"
add CM5b_ep24 "$(run_file CM5b_carphase_trilinear_lr6e-4_cap1M checkpoint_final.pt)"
add M5c_ep24 "$(run_file M5c_trilinear_lr6e-4_target0.5_1024k checkpoint_final.pt)"
add CM5c_ep24 "$(run_file CM5c_carphase_trilinear_lr6e-4_target0.5_cap1M checkpoint_final.pt)"
add T5c_ep24 "$(run_file T5c_trilinear_lr6e-4_target0.5 checkpoint_final.pt)"
add I5_ep24 "$(run_file I5_inherit_lr6e-4 checkpoint_final.pt)"
add D2_ep10 $S/D2_ep10_pre.pt; add D2_ep24 $S/D2_ep24_pre.pt; add D2_ep40 "$(run_file D2_sh2_trilinear_lr6.7e-5_ep40 checkpoint_final.pt)"
for e in 10 16 24; do add F5full_ep$e $S/F5full_ep${e}_pre.pt; done
add F5full_now "$(run_file F5_full_v2_cap1M_ep40 checkpoint_latest.pt)"
add F5c_now "$(run_file F5c_full_v2_cap1M_ep40_target0.5 checkpoint_latest.pt)"
A_EXT="$(run_file F5gA_ext40 checkpoint_latest.pt)"
add F5gA_now "${A_EXT:-$(run_file F5gA_pooled_warmstart_ep12 checkpoint_latest.pt)}"
add F5gB_now "$(run_file F5gB_learnable_gain_ep12 checkpoint_latest.pt)"
add F5gD_now "$(run_file F5gD_curvature_start_ep12 checkpoint_latest.pt)"
add CM5bx70_now "$(run_file CM5b_ext70 checkpoint_latest.pt)"
add CM5cx70_now "$(run_file CM5c_ext70 checkpoint_latest.pt)"
add D2x70_now "$(run_file D2_ext70 checkpoint_latest.pt)"
RUNS+=("data_raw=field:$DI/raw_all.npz:data_energy" "data_v2=field:$DI/carphase_v2_all.npz:data_energy")
STAMP=$(date +%Y%m%d_%H%M)
EXPORT=$D/pointclouds/viewer_$STAMP
JOB=$D/pointclouds/export_$STAMP.sbatch
{
  echo '#!/bin/bash'
  echo '#SBATCH --job-name=gotcha-viewer-export --account=158648339640 --partition=cpu --qos=normal --exclude=ac091'
  echo "#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --output=$D/pointclouds/export-$STAMP-%j.log"
  echo 'set -eo pipefail; cd /home/u.db364833/RIFT; source .local-setup/activate-pvc.sh'
  echo 'export PYTHONPATH=$PWD RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8'
  printf 'python -u scripts_pvc/export_gotcha_rift_pointcloud_pvc.py --geometry-dir %s --out-dir %s' "$D/geometry" "$EXPORT"
  for r in "${RUNS[@]}"; do printf ' --run %q' "$r"; done
  echo
} > $JOB
echo "$(date +%H:%M:%S) export of ${#RUNS[@]} sets -> $EXPORT"
sbatch --wait $JOB
source .local-setup/activate-pvc.sh >/dev/null 2>&1
python scripts_pvc/build_gotcha_rift_viewer.py $EXPORT "$OUT_HTML"
echo "$(date +%H:%M:%S) built $OUT_HTML"
