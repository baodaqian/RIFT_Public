#!/bin/bash
# SE SPGL1 Stage 1 for GOTCHA Camry (scripts_pvc/se_spgl1_stage1.py solve|assemble --gotcha) on the CPU partition.
# No GPU: does not count against the PVC GRES cap. The GOTCHA arguments are the C3 production
# Camry SE command's (cap16/stride2, passes 1-8, HH, 1500 train, se_g40_readout48.json), planned
# by train_gotcha_dataset_pvc.py itself; only the output directory differs.
# Dense Eq. 2 operators are 24-116 GB per sub-aperture (complex128, 23k-113k rows x 64000 voxels),
# hence the memory request; sub-apertures run one at a time inside a job.
# Usage: export SE_COMMAND=solve SE_GROUPS=0-11 SE_ROOT=/scratch/.../se_spgl1_<date>
#        sbatch --export=ALL scripts_pvc/se_spgl1_stage1_gotcha_cpu.sbatch
# SE_COMMAND=assemble (SE_GROUPS unset) once all 72 sub-apertures exist.
#SBATCH --account=158648339640
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=06:00:00
set -eo pipefail
cd /home/u.db364833/RIFT
source .local-setup/activate-pvc.sh
set -u
export PYTHONPATH="$PWD" RIFT_ACCELERATOR=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
C3=/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260922_camry_batched_4jobs
OUT=$SE_ROOT/outputs/gotcha/camry/f5d77de24a59fa0b/pulse_subset16_all_roles_v2/frequency_stride2_all_roles_v2/sugavanam_ertin_spgl1
python -u scripts_pvc/se_spgl1_stage1.py "$SE_COMMAND" --output "$OUT" ${SE_GROUPS:+--groups $SE_GROUPS} \
  --gotcha --dataset-root /scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined \
  --region camry --polarizations hh --passes 1 2 3 4 5 6 7 8 --num-train 1500 --num-tx 1 --num-rx 1 \
  --pulses-per-sector 16 --frequency-stride 2 --method sugavanam_ertin \
  --config $C3/source/protocols/se_g40_readout48.json
