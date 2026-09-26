#!/bin/bash
# Six-method renders + fixed-threshold 3D metrics for one RIFT-dataset scene (CPU partition).
# Usage: export OBJECT=a320 SCENE=airliner_a320 OUT=.../geometry [SKIP="geraf ..."]   (se_stage1: SE Stage-1 diagnostic)
#        [BP=.../backprojection]   adds the matched-filter backprojection $BP/$OBJECT/matched_filter.npz
#        sbatch --export=ALL --job-name postflight-$OBJECT --output $OUT/$OBJECT-%j.log scripts_pvc/run_rift_dataset_six_method_postflight_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -eo pipefail
: "${OBJECT:?}" "${SCENE:?}" "${OUT:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu RIFT_PVC_TCNN_SHIM=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
G=/scratch/group/p.cis261724.000/RIFT_pvc_runs; B=outputs/rift/train2400/1t1r_2010b2bbe725
P=$G/production_20260921_35jobs/$B/$SCENE; Q=$G/production_20260921_rf_rs_14jobs/$B/$SCENE
S8=$G/production_20260922_radarsplat_logpower_7jobs/$B/$SCENE; SE=$G/se_spgl1_20260922/$B/$SCENE/sugavanam_ertin_spgl1
ARGS=(--object "$OBJECT" --scene "$SCENE" --role-manifest "$P/role_manifest.json" --output-dir "$OUT/$OBJECT")
skip() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
skip rift || ARGS+=(--rift "$P/rift/checkpoint_best.pth.tar")
skip spinr || ARGS+=(--spinr "$P/spinr/budget48-direct-150/checkpoint_best.pth.tar")
skip geraf || ARGS+=(--geraf "$P/geraf/source_v1/checkpoints/checkpoint_best.pth.tar")
skip radar_fields || ARGS+=(--radar-fields "$Q/radar_fields/checkpoint_best.pth.tar")
skip radarsplat || ARGS+=(--radarsplat "$S8/radarsplat/budget48/checkpoints/geometry_g48.npz")
skip se || ARGS+=(--se-surface "$SE/surface.npz" --se-checkpoint "$SE/checkpoint_final.pt")
skip se_stage1 || ARGS+=(--se-stage1 "$SE/stage1_selected.pt")
if [ -n "${BP:-}" ] && ! skip backprojection; then ARGS+=(--backprojection "$BP/$OBJECT/matched_filter.npz"); fi
hostname; date
python -u scripts_pvc/run_rift_dataset_six_method_postflight_pvc.py "${ARGS[@]}"
date
