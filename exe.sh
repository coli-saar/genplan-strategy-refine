#!/usr/bin/env bash

source /scratch/aklaus/projects/autoplanbench/genplan/rename_gpus.sh

source /nethome/aklaus/miniconda3/etc/profile.d/conda.sh

conda activate autoplan

echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

export CUDA_VISIBLE_DEVICES=0 #,1,2,3

export OLLAMA_MODELS=/scratch/aklaus/ollama_models
# export OLLAMA_CONTEXT_LENGTH=8192

echo "using port $1"

export OLLAMA_HOST=127.0.0.1:$1

OLLAMA_LOAD_TIMEOUT=15m
/scratch/aklaus/ollama/bin/ollama serve &
/scratch/aklaus/ollama/bin/ollama ps

# /scratch/aklaus/ollama/bin/ollama pull alibayram/Qwen3-30B-A3B-Instruct-2507
# /scratch/aklaus/ollama/bin/ollama run alibayram/Qwen3-30B-A3B-Instruct-2507 "Write a short poem about the king of snails"

echo "now starting"

# python create_sh_scripts.py
# bash sh_scripts_generation/additional-ferry_test_generation.sh 
# bash sh_scripts_evaluation/additional-ferry_test_evaluation.sh

bash sh_scripts_$2/$3_$2.sh

