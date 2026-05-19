#!/bin/bash
#SBATCH -J ray-smoke
#SBATCH -o slurm-ray-smoke-%j.out
#SBATCH -e slurm-ray-smoke-%j.err
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
export RAY_TMPDIR=/tmp/ray_hydro
export HYDRO_SMOKE=1
cd /work/10655/ejokhan123/ls6/hydro-finetune
python src/training/ray_tune_qlora.py
