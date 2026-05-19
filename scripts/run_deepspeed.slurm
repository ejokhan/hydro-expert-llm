#!/bin/bash
#SBATCH -J ds-ftune
#SBATCH -o slurm-deepspeed-%j.out
#SBATCH -e slurm-deepspeed-%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=0
#SBATCH -t 06:00:00
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

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR

cd /work/10655/ejokhan123/ls6/hydro-finetune

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/training/deepspeed_finetune.py
