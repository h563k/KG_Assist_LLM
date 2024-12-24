#!/usr/bin/env bash

home="/opt/project/llama-factory"
CONDA_PATH="/root/miniconda3"
ENV_NAME="llama_fact"


# 激活conda环境
. "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 使用date命令并格式化输出，精确到日期
current_time=$(date +"%Y-%m-%d")

cd "$home/LLaMA-Factory"

# 进程清理
ps aux | grep llamafactory | awk '{print $2}' | xargs kill
rm -rf "$home/logs/$current_time.log"
mkdir -p "$home/logs"

export CUDA VISIBLE DEVICES=0,1
nohup llamafactory-cli train "$home/setup_files/llama3_lora_sft.yaml">"$home/logs/$current_time.log" 2>&1 &
echo "start successful"