#!/bin/bash
# SE SPGL1 Stage 1 (scripts_pvc/se_spgl1_stage1.py solve|assemble) on the CPU partition.
# No GPU: does not count against the PVC GRES cap.
# Usage: export SCENE=supersonic_x59 SE_COMMAND=solve SE_GROUPS=0-35 SE_ROOT=/scratch/.../se_spgl1_<date>
#        sbatch --export=ALL scripts_pvc/se_spgl1_stage1_cpu.sbatch
# SE_COMMAND=assemble runs the assembly and validation readout once all groups exist.
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=03:00:00
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
C1=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs
MANIFEST=$C1/outputs/rift/train2400/1t1r_2010b2bbe725/$SCENE/role_manifest.json
OUT=$SE_ROOT/outputs/rift/train2400/1t1r_2010b2bbe725/$SCENE/sugavanam_ertin_spgl1
python -u scripts_pvc/se_spgl1_stage1.py "$SE_COMMAND" \
  --npz-path /scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset/objects/$SCENE.npz \
  --parent-role-manifest "$MANIFEST" \
  --config $C1/source/protocols/se_g40_readout48.json \
  --output "$OUT" ${SE_GROUPS:+--groups $SE_GROUPS}
