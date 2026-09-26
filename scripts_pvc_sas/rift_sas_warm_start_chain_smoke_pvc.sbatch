#!/bin/bash
# Smoke test of the chained CGLS warm start and the renderer override (docs/RIFT_SAS_Train.md, A29) on one PVC card.
# Test-only fit set (--max-fit-pings 16), outputs under rift_sas_train/smoke_chain_<job>/; nothing else is touched.
#   A  fresh, 2 iterations                       -> cgls_state.pt at iteration 2
#   B  --continue-state A, 1 iteration           -> iteration 3
#   C  fresh, 3 iterations                       -> reference for B (exact continuation: same x up to XPU nondeterminism)
#   D  --restart-from A's step-0 checkpoint, 1   -> the restart check reproduces A's recorded fit; iteration 3
#   E  --restart-from A with --fit-stride 16     -> must refuse (fit set differs)
#   F  read-out of A's step 0 with the override lambertian 1 / opacity 0, and the split driver on it (4 pings)
#   sbatch scripts_pvc_sas/rift_sas_warm_start_chain_smoke_pvc.sbatch
# SMOKE_EXIT=0: every check passed, 0 fallback lines; 1 otherwise.
#SBATCH --job-name=pvc-sas-chain-smoke
#SBATCH --account=158648339640
#SBATCH --partition=pvc
#SBATCH --qos=normal
#SBATCH --exclude=ac023
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG/logs/%x-%j.log
set -o pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
unset RIFT_ACCELERATOR RIFT_PVC_ALLOW_BACKEND   # never inherit a forced backend from the submitting shell
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_DEBUG_XPU_FALLBACK=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rift-mpl-${SLURM_JOB_ID}" XDG_CACHE_HOME="${TMPDIR:-/tmp}/rift-cache-${SLURM_JOB_ID}"
source .env.datasets.sh
PKG=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageG
LOG=$PKG/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log
CACHE="${AIRSAS_CACHE_ROOT:?}/armadillo_5khz_trainonly_120rings_v1_20260912"
OUT=$PKG/rift_sas_train/smoke_chain_${SLURM_JOB_ID}
mkdir -p "$OUT" || exit 1
hostname; date; python -c "from rift_pvc import accelerator as a; print(a.describe())"

argv() {  # the pilot argv (rift_sas_pilot_pvc.sbatch) for run name $1
  echo --cache "$CACHE" --model adaptive_rift_sas --checkpoint-root "$OUT" --checkpoint-name "$1" \
    --require-explicit-splits --sh-degree 3 --num-rays 4900 --max-bins 110 --grad-clip 1.0 \
    --beamwidth-deg 30 --sh-direction rx_to_point --seed 42 --profile full \
    --initial-granularity 64 --granularity 128 --adaptive-capacity 262144 --max-active 262144 \
    --refine-every 1000000000 --probe-every 1000000000 --checkpoint-every 500 \
    --coefficient-lr 1e-4 --position-lr 1e-4 --init-scale 1e-2 --lambertian-ratio 0.0 --opacity-scale 500
}
warm() {  # warm() <name> <warm-start options...>
  local name=$1; shift
  echo "=== $name: $* ==="
  python -u scripts_pvc_sas/rift_sas_warm_start.py --fit-stride 8 --max-fit-pings 16 --time-budget 0 "$@" -- $(argv "$name")
}
rc=0
warm A --iterations 2 || rc=1
warm B --iterations 1 --continue-state "$OUT/A/cgls_state.pt" || rc=1
warm C --iterations 3 || rc=1
warm D --iterations 1 --restart-from "$OUT/A/checkpoint_latest.pt" || rc=1
echo "=== E: must refuse ==="
if python -u scripts_pvc_sas/rift_sas_warm_start.py --fit-stride 16 --max-fit-pings 16 --iterations 1 --time-budget 0 \
     --restart-from "$OUT/A/checkpoint_latest.pt" -- $(argv E) 2>&1 | tee "$OUT/E.txt"; then
  echo "E did not refuse"; rc=1
elif grep -q "fit-stride is 8, this leg has 16\|the fit set differs" "$OUT/E.txt"; then
  echo "E refused as required"
else
  echo "E failed for another reason"; rc=1
fi

echo "=== checks on the CGLS states ==="
python - "$OUT" <<'EOF' || rc=1
import json, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
load = lambda name: torch.load(out / name / "cgls_state.pt", map_location="cpu", weights_only=False)
A, B, C, D = (load(n) for n in "ABCD")
ok = True
def check(label, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + label); ok &= bool(cond)
check(f"iterations A/B/C/D = {A['iteration']}/{B['iteration']}/{C['iteration']}/{D['iteration']} (2/3/3/3)",
      (A["iteration"], B["iteration"], C["iteration"], D["iteration"]) == (2, 3, 3, 3))
check("B history carries A's two rows", [r["iteration"] for r in B["history"]] == [1, 2, 3]
      and B["history"][:2] == A["history"])
diff = max(float((b - c).norm() / c.norm()) for b, c in zip(B["x"], C["x"]))
check(f"B (2 + continued 1) equals C (fresh 3): max rel |x_B - x_C| = {diff:.2e} (< 1e-3)", diff < 1e-3)
fits = [r["fit_rel_mse"] for r in C["history"]]
check(f"fit falls over C's iterations {['%.5f' % f for f in fits]}", all(b < a for a, b in zip(fits, fits[1:])))
print(f"B fit {B['history'][-1]['fit_rel_mse']:.6f}, C fit {C['history'][-1]['fit_rel_mse']:.6f}, "
      f"D (restart) fit {D['history'][-1]['fit_rel_mse']:.6f}; B residual drift {B['continuation']['residual_drift']:.2e}")
rc_ = D["continuation"]["restart_check"]
check(f"D restart check: fit at x0 {rc_['fit_rel_mse']:.6f} vs A's recorded {rc_['recorded_fit_rel_mse']:.6f}",
      abs(rc_["fit_rel_mse"] - rc_["recorded_fit_rel_mse"]) < 1e-3)
check(f"D held-out at x0 reproduces A's iteration 2: TRAIN64 in-band {rc_['train64_in_band_g1']:.5f} vs "
      f"{A['history'][-1]['train64_in_band_g1']:.5f}", abs(rc_["train64_in_band_g1"] - A["history"][-1]["train64_in_band_g1"]) < 1e-3)
check("D improves on A's fit", D["history"][-1]["fit_rel_mse"] < A["history"][-1]["fit_rel_mse"])
for n in "BCD":
    ws = json.loads((out / n / "warm_start.json").read_text())
    check(f"{n}: warm_start.json cgls_iterations_this_leg {ws['cgls_iterations_this_leg']}, continuation "
          f"{(ws['cgls_continuation'] or {}).get('mode')}", ws["cgls_iterations_this_leg"] >= 1)
sys.exit(0 if ok else 1)
EOF

echo "=== F: renderer override and split driver ==="
python -u scripts_pvc_sas/rift_sas_checkpoint_readout.py "$OUT/A/checkpoint_latest.pt" --device xpu --train-pings 4 --val-pings 4 \
  --lambertian-ratio 1 --opacity-scale 0 --json "$OUT/F_override.json" | tee "$OUT/F.txt" || rc=1
grep -q "renderer OVERRIDE: lambertian_ratio 1 opacity_scale 0" "$OUT/F.txt" || { echo "F: no override line"; rc=1; }
python -u scripts_pvc_sas/rift_sas_renderer_split.py "$OUT/A/checkpoint_latest.pt" --out-dir "$OUT/split" --device xpu --pings 4 || rc=1
python - "$OUT" <<'EOF' || rc=1
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
P = json.loads((out / "split" / "ckpt0_P.json").read_text())
own = json.loads((out / "F_override.json").read_text())
L = json.loads((out / "split" / "ckpt0_L.json").read_text())
ok = (P["renderer"]["overridden"] is False and L["renderer"]["overridden"] is True
      and abs(L["train"]["12.5..27.5"]["rel_mse_g_train_fit"] - own["train"]["12.5..27.5"]["rel_mse_g_train_fit"]) < 1e-6)
print(("PASS" if ok else "FAIL") + ": split P not overridden, L overridden and equal to the direct override read-out")
sys.exit(0 if ok else 1)
EOF
date
n=$(grep -c 'fallback from XPU to CPU' "$LOG" || true); echo "XPU->CPU fallback lines in this log: $n"
[ "$n" != "0" ] && rc=1
echo "SMOKE_OUT=$OUT"
echo "SMOKE_EXIT=$rc"; exit $rc
