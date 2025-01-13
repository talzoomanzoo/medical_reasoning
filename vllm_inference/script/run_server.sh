#!bin/bash
CUDA_VISIBLE_DEVICES=$1
MODEL_NAME_OR_PATH=$2
TENSOR_PARALLEL_SIZE=$3

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server --model $MODEL_NAME_OR_PATH --tensor-parallel-size $TENSOR_PARALLEL_SIZE --seed 42 --port $4