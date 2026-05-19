#!/bin/bash
#SBATCH -J ds-smoke
#SBATCH -o slurm-ds-smoke-%j.out
#SBATCH -e slurm-ds-smoke-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -A TG-NAIRR240491

source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag

export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export HF_TOKEN=$(python -c "from huggingface_hub import get_token; print(get_token())")
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/opt/apps/cuda/12.2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Verbose NCCL during smoke so we can see exactly what transport it picks
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR

export HYDRO_SMOKE=1

cd /work/10655/ejokhan123/ls6/hydro-finetune

torchrun \
    --nproc_per_node=3 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/training/deepspeed_finetune.py
