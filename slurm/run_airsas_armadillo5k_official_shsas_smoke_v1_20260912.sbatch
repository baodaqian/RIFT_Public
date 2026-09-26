#!/bin/bash
# AirSAS Armadillo5kHz official SH-SAS smoke launcher.
# The research script owns smoke validation and completion evidence.
# This launcher is staged locally for manager-owned review; it does not submit itself.
#SBATCH --job-name=mgr_rift_airsas_arm5k_official_shsas_smoke_v1_20260912
#SBATCH --account=gts-jromberg3-ece
#SBATCH --qos=inferno
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=12G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude=atl1-1-01-004-31-0,atl1-1-03-007-29-0,atl1-1-01-002-8-0,atl1-1-02-006-34-0

set -euo pipefail
umask 077

exec bash /storage/project/r-jromberg3-0/dbao31/RIFT/scripts/run_official_shsas_smoke.sh
