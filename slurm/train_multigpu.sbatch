#!/bin/bash
#SBATCH --job-name=RIFTmg                      # Job name
#SBATCH --account=gts-jromberg3-coda20         # charge account
#SBATCH --export=NONE
#SBATCH -N1 --gres=gpu:H200:4                  # ONE node, 4 GPUs (override at submit time)
#SBATCH -C gpu-h200
#SBATCH --ntasks-per-node=1                    # one torchrun launcher; it forks one proc per GPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G                             # every rank loads the npz independently (see below)
#SBATCH -t3-00:00:00                           # Duration of the job
#SBATCH -qinferno                              # QOS Name
#SBATCH -oReport-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail preferences
#SBATCH --mail-user=dbao31@gatech.edu          # E-mail address for notifications
#
# SCENE-SHARDED multi-GPU training: the same train.py, launched under
# torchrun with one process per GPU, each owning 1/N of the voxels. The
# radar forward operator is a sum over scatterers, so summing the ranks'
# partial renders is exact -- this is the SAME experiment as a single-GPU
# run, just with ~N times the scene-memory budget and ~N times the
# per-viewpoint throughput. See rift/distributed.py for the design and
# scripts/validate_distributed_scene.py for the correctness gate.
#
# Submit exactly like slurm/train.sbatch -- all train.py flags pass through:
#
#   sbatch --gres=gpu:H200:8 slurm/train_multigpu.sbatch \
#       --data-format npz --npz-path data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz \
#       --scene-repr grid_sh --forward-operator range --compute-dtype float64 \
#       --checkpoint-name b787_dense_g96_8gpu --granularity 96 --extent 0.15 \
#       --point-chunk 16384 --num-train 1800 ...
#
# GPU COUNT is set by --gres at submit time (the directive above is only a
# default); the script reads it back from SLURM and launches that many ranks.
# Multi-GPU currently requires --scene-repr grid_sh (train.py errors out
# otherwise).
#
# MEMORY: this is deliberately single-node. Every rank np.load()s the whole
# npz response array and builds its own per-view magnitude/phase tensors, so
# host RAM scales with N_GPU * (npz size + views*nf*Tx*Rx*2*4B) -- ~5 GB per
# rank for the 2.46 GB B787 sphere2k npz at 1800 views. 256G covers 8 ranks
# with headroom; raise it for the 24 GB sphere/cube npz files.
#
cd /storage/project/r-jromberg3-0/dbao31/RIFT

module purge
module load anaconda3/2023.03

CONDA_BASE=/usr/local/pace-apps/manual/packages/anaconda3/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate RIFT
export PYTHONUNBUFFERED=1

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
# torchrun defaults OMP_NUM_THREADS to 1 with a warning; give each rank a
# fair share of the allocated cores instead (the npz load and the host-side
# dataset build are the only CPU-heavy parts).
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

echo "RIFT scene-sharded training on ${NGPU} GPU(s), OMP_NUM_THREADS=${OMP_NUM_THREADS}: $@"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" train.py "$@"

echo "RIFT training finished"
