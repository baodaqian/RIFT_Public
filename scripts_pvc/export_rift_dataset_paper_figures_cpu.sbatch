#!/bin/bash
# Manuscript view-figure assets (method MIP panels, reference-mesh views, colour bar) on the CPU partition.
# Usage: export PAPER_DIR=manuscripts/iclr27/figures/rift_dataset [FALLBACK=geometry_nogeraf]
#        sbatch --export=ALL --output .../paper-figs-export-%j.log scripts_pvc/export_rift_dataset_paper_figures_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --exclude=ac091
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=paper-figs-export
set -eo pipefail
: "${PAPER_DIR:?}"
cd "${SOURCE_ROOT:-/home/u.db364833/RIFT}"
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu RIFT_PVC_ALLOW_BACKEND=cpu MPLBACKEND=Agg PYTHONUNBUFFERED=1
ARGS=(--paper-dir "$PAPER_DIR")
for root in ${FALLBACK:-}; do ARGS+=(--fallback "$root"); done
hostname; date
python -u scripts_pvc/export_rift_dataset_paper_figures_pvc.py "${ARGS[@]}"
date
