#!/bin/bash
# SE Stage-1 SPGL1 pilot (scripts_pvc/se_spgl1_pilot.py) on the CPU partition.
# No GPU: does not count against the PVC GRES cap. Reads production checkpoints only.
# Usage: export SCENE=supersonic_x59 SE_GROUPS=0 OUT=/scratch/.../se_spgl1_pilot_20260922/x59
#        sbatch --export=ALL scripts_pvc/se_spgl1_pilot_cpu.sbatch
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=06:00:00
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
# SPGL1: https://github.com/drrelyea/spgl1 at 405ca805a2d56d783a0445e834c801c5b7c2263a (clone, not installed into the env)
export PYTHONPATH="$PWD:/scratch/group/p.cis261724.000/envs/se_spgl1/src" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
C1=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs
RUN=$C1/outputs/rift/train2400/1t1r_2010b2bbe725/$SCENE
python -u scripts_pvc/se_spgl1_pilot.py \
  --npz-path /scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset/objects/$SCENE.npz \
  --parent-role-manifest $RUN/role_manifest.json \
  --config $C1/source/protocols/se_g40_readout48.json \
  --checkpoint $RUN/sugavanam_ertin/checkpoint_final.pt \
  --groups "$SE_GROUPS" --output "$OUT" ${PILOT_ARGS:-}
