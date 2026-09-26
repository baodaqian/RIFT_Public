#!/bin/bash
# Matched B787 geometry audit for every completed Round-8 Stage-O checkpoint.
# Resource/QoS/output routing is supplied by the experiment manager or sbatch.
set -euo pipefail

RIFT_ROOT=/storage/project/r-jromberg3-0/dbao31/RIFT
RUN_ROOT=/storage/scratch1/1/dbao31/rift_round8_runs/20260810
OUT_ROOT=/storage/scratch1/1/dbao31/rift_round8_runs/20260810/evaluation/plenoxel_trilinear_geometry_v1
PYTHON=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/RIFT/bin/python

cd "$RIFT_ROOT"
mkdir -p "$OUT_ROOT"

checkpoints=(
  "$RUN_ROOT/b787_r8o_legacy_s42/checkpoint_best.pth.tar"
  "$RUN_ROOT/b787_r8o_e15_lr3em6_s42/checkpoint_best.pth.tar"
  "$RUN_ROOT/b787_r8o_e15_lr3em5_s42/checkpoint_best.pth.tar"
  "$RUN_ROOT/b787_r8o_e15_lr3em4_s42/checkpoint_best.pth.tar"
  "$RUN_ROOT/b787_r8o_e15_lr3em3_s42/checkpoint_best.pth.tar"
  "training_checkpoints/b787_r6_occ_dc/checkpoint_best.pth.tar"
  "training_checkpoints/b787_r7_target20k_shdeg1em9/checkpoint_final.pth.tar"
)
labels=(r8_legacy r8_e15_lr3em6 r8_e15_lr3em5 r8_e15_lr3em4 r8_e15_lr3em3 \
        rift_r6_learned_dc rift_r7_target20k_shdeg1em9)

for checkpoint in "${checkpoints[@]}"; do
  test -s "$checkpoint"
done

"$PYTHON" scripts/eval_b787_geometry_metrics.py \
  --checkpoints "${checkpoints[@]}" \
  --labels "${labels[@]}" \
  --upsample 1 \
  --csv "$OUT_ROOT/native_grid_rows.csv.tmp"
mv -f "$OUT_ROOT/native_grid_rows.csv.tmp" "$OUT_ROOT/native_grid_rows.csv"

"$PYTHON" scripts/eval_b787_geometry_metrics.py \
  --checkpoints "${checkpoints[@]}" \
  --labels "${labels[@]}" \
  --upsample 4 \
  --mesh-variant \
  --min-points 1280 \
  --csv "$OUT_ROOT/trilinear_4x_rows.csv.tmp"
mv -f "$OUT_ROOT/trilinear_4x_rows.csv.tmp" "$OUT_ROOT/trilinear_4x_rows.csv"

"$PYTHON" scripts/render_recon_grid.py \
  --set round8 \
  --checkpoint-root "$RUN_ROOT" \
  --upsample 4 \
  --planes 0,2 \
  --out "$OUT_ROOT/r8_trilinear_4x_top_front.png"

"$PYTHON" scripts/summarize_r8_geometry.py \
  --out-root "$OUT_ROOT" \
  --reference-csv figures/b787_recon/rift_rounds_5_6_7_geometry.csv
