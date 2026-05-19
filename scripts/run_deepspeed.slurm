#!/bin/bash
#SBATCH -J ds-ftune
#SBATCH -o slurm-deepspeed-%j.out
#SBATCH -e slurm-deepspeed-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -A TG-NAIRR240491

source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag

# --- Standard env ---
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export HF_TOKEN=$(python -c "from huggingface_hub import get_token; print(get_token())")
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/opt/apps/cuda/12.2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# --- NCCL transport config for single-node A100-PCIE on RHEL 8 kernel ---
# IB: disabled — no point on a single node, and it was the source of the segfault
# P2P: disabled — A100-PCIE + kernel 4.18 + IOMMU causes eager_connect SIGSEGV
# SHM: enabled (default) — shared memory is the right intra-node transport here
# Net interface: loopback for any control-plane sockets
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

# --- Triton cache off NFS (silences the warning, avoids potential hangs) ---
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR

cd /work/10655/ejokhan123/ls6/hydro-finetune

# Launcher: torchrun (not `deepspeed`). HF Trainer's deepspeed= config arg
# works with either, but torchrun respects the NCCL env vars more reliably
# and uses torch's standard init path.
torchrun \
    --nproc_per_node=3 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/training/deepspeed_finetune.py
