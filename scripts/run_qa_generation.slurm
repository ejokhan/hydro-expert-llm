#!/bin/bash
#SBATCH -J qa-gen
#SBATCH -o slurm-qa-gen-%j.out
#SBATCH -e slurm-qa-gen-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH -A TG-NAIRR240491

source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd /work/10655/ejokhan123/ls6/hydro-finetune

python src/data_prep/generate_qa_gpu.py
