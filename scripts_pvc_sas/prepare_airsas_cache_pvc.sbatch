#!/bin/bash
# G4 of the sonar PVC port: build the shared AirSAS5k cache on ACES with the
# UNCHANGED scripts/prepare_airsas_cache.py (numpy only) from the staged Reed
# frontend outputs and system data (prerequisite P0), then validate it.
# Inputs (environment, see .env.datasets.sh):
#   AIRSAS_FRONTEND_ROOT   parent of airsas_<scene>5k_frontend_trainonly_v1_<date>/
#   AIRSAS_SYSTEM_DATA_ROOT directory holding system_data_arma_5k.pik / system_data_bunny_5k.pik
#   AIRSAS_CACHE_ROOT      destination parent for the caches
#   REED_ROOT              the Reed_SAS_reference checkout (data_schemas.py for the restricted unpickler)
#   SCENE=armadillo|bunny  (default armadillo)
#SBATCH --job-name=pvc-prepare-airsas-cache
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
SCENE=${SCENE:-armadillo}
: "${AIRSAS_FRONTEND_ROOT:?}" "${AIRSAS_SYSTEM_DATA_ROOT:?}" "${AIRSAS_CACHE_ROOT:?}" "${REED_ROOT:?}"
case "$SCENE" in
  armadillo) FRONTEND=airsas_armadillo5k_frontend_trainonly_v1_20260912; PIK=system_data_arma_5k.pik
             CACHE=armadillo_5khz_trainonly_120rings_v1_20260912 ;;
  bunny)     FRONTEND=airsas_bunny5k_frontend_trainonly_v1_20260913; PIK=system_data_bunny_5k.pik
             CACHE=bunny_5khz_trainonly_120rings_v1_20260913 ;;
  *) echo "SCENE must be armadillo or bunny"; exit 2 ;;
esac
hostname; date
python -u scripts/prepare_airsas_cache.py --scene "$SCENE" \
  --system-data "$AIRSAS_SYSTEM_DATA_ROOT/$PIK" \
  --weights-dir "$AIRSAS_FRONTEND_ROOT/$FRONTEND/deconvolved_measurements/numpy" \
  --output "$AIRSAS_CACHE_ROOT/$CACHE" --reed-root "$REED_ROOT" --bandwidth-khz 5 --normalization-mode train_only
rc=$?; [ $rc -ne 0 ] && { echo "PREPARE_EXIT=$rc"; exit $rc; }
python scripts/validate_airsas_5k_cache.py "$AIRSAS_CACHE_ROOT/$CACHE"; rc=$?
python - "$AIRSAS_CACHE_ROOT/$CACHE/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print({k: m.get(k) for k in ("dataset_identity", "frontend_asset_identity", "num_pings", "num_bins", "original_num_bins",
                             "ring_size", "num_rings", "ring_protocol")}, m["split_contract"]["counts"], m["frontend_normalization"])
PY
date; echo "PREPARE_EXIT=$rc"; exit $rc
