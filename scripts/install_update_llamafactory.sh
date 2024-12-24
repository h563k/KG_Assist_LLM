#!/usr/bin/env bash
# conda create -n llama_fact python=3.11
# export PYTHONPATH=$PYTHONPATH:'pwd'

CONDA_PATH=/root/miniconda3
ENV_NAME=llama

. "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate $ENV_NAME

cd /opt/project/KG_Assist_LLM/data/LLaMA-Factory-0.9.1
git pull
pip install -e ".[torch,metrics]"
