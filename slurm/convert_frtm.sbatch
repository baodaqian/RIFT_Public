#!/bin/bash
#SBATCH --job-name=RIFT-frtm-convert           # Job name
#SBATCH --account=gts-jromberg3-coda20         # charge account
#SBATCH --export=NONE
#SBATCH -N1 -n1 -c4 --mem=16G                  # CPU-only, no GPU needed for .frtm parsing
#SBATCH -t0-02:00:00                           # Duration of the job (conversion validated at ~1.1s/file, 795 files)
#SBATCH -qinferno                              # QOS Name
#SBATCH -oReport-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail preferences
#SBATCH --mail-user=dbao31@gatech.edu          # E-mail address for notifications
#
# Converts data/ansysRCS/B787_50m/zone1-5 raw .frtm files to per-viewpoint
# training CSVs at data/AEDT_B787_Sample/zone{N}_CSV. See
# scripts/convert_frtm_to_csv.py for the conversion logic (validated on a
# 3-file sample before this job was submitted) and
# EXPERIMENT_MANAGER_HANDOFF.md for the radar-params mismatch found during
# validation (frequency band and Rx count differ from rift/config.py --
# does NOT block this conversion step, but blocks training on the result
# until resolved).
#
cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT

echo "B787 .frtm -> CSV conversion starting"
python scripts/convert_frtm_to_csv.py "$@"
echo "B787 .frtm -> CSV conversion finished"
