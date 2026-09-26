# Copy to .env.datasets.sh and replace every /path/to/... value.
export RIFT_DATA_ROOT=/path/to/RIFT_dataset
export GOTCHA_DATA_ROOT=/path/to/GOTCHA-CP_Combined
export RIFT_RUN_ROOT=/path/to/persistent/scratch/rift-runs
export GOTCHA_RUN_ROOT=/path/to/persistent/scratch/gotcha-runs

# Optional sonar inputs for the retained PVC sonar tools.
export AIRSAS_CACHE_ROOT=/path/to/prepared/airsas/cache
export AIRSAS_SYSTEM_DATA_ROOT=/path/to/airsas/system_data_files
export AIRSAS_FRONTEND_ROOT=/path/to/airsas/frontend_outputs
export SH_SAS_REED_ROOT=/path/to/sonar/data
export REED_ROOT=/path/to/Neural-Volumetric-Reconstruction-for-Coherent-SAS
