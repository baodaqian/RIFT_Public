#!/bin/bash
# Card smoke of GOTCHA GeRaF with the configured light_power start
# (protocols/geraf_mf48_gotcha.json: light_power_start=train_warm_start_16_views, trans_power=1,
# receiver_geometry=float64_intersection): the Camry production command stopped by SIGTERM after
# 25 min. The pre-warm-start reference is smoke 2156223 (validation RelMSE 1.0000 at steps
# 1000/2000). Any "Aten Op fallback" line is a defect.
# Any "Aten Op fallback" line is a defect.
#SBATCH --job-name=pvc-smoke-gotcha-geraf-warm-start
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/geraf_warm_start_smoke/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
OUT=/scratch/group/p.cis261724.000/RIFT_pvc_runs/geraf_warm_start_smoke/outputs_${SLURM_JOB_ID}
hostname; date; git -C "$PWD" rev-parse HEAD; git -C "$PWD" status --short rift/geraf_source.py rift_pvc/geraf_source.py
timeout -s TERM -k 300 1500 python -W always -u train_gotcha_dataset_pvc.py \
  --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined --output-root "$OUT" \
  --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 16 --frequency-stride 2 --method geraf --device xpu --config protocols/geraf_mf48_gotcha.json 2>&1 \
  | grep -v '^ *"\|^ *[]}{[]\+,\?$\|^ *-\?[0-9.e+-]\+,\?$'
rc=$?; date; echo "SMOKE_EXIT=$rc (124 or 143 = stopped by the 25-minute bound)"
find "$OUT" -name light_power_warm_start.json -exec cat {} ;
find "$OUT" -name validation.json -exec python -c "import json,sys;[print({k:r[k] for k in ('step','mf_relative_mse','native_complex_relative_mse')}) for r in json.load(open(sys.argv[1]))]" {} ;
find "$OUT" -name source_run.json -exec python -c "import json,sys;r=json.load(open(sys.argv[1]))['recipe'];print('recipe:',r.get('receiver_geometry'),r.get('light_power_start'),r['trans_power'])" {} ;
exit $rc
