#!/usr/bin/env bash

export PORT=18081

num_prompts=1000
ranges=("32" "128" "512" "1536")
backend="vllm"
model_name="/data/Llama-2-13b-hf"
result_dir="./benchmark_llama13b"
# model_name="/data/Llama-2-70b-chat-hf"
# result_dir="./benchmark_llama70b"

# Run real test
for i in ${!ranges[@]}; do
    range=${ranges[$i]}
    echo "range $range"

    ulimit -n 65536 && python3 benchmark_llm_serving.py \
            --backend $backend \
            --tokenizer $model_name \
            --port $PORT \
            --random_prompt_lens_mean 512 \
            --random_prompt_lens_range 0 \
            --num_requests $num_prompts \
            --stream True \
            --allow_random_response_lens \
            --random_response_lens_mean 128 \
            --random_response_lens_range $range \
            --response_distribution capped_exponential \
            --allow_random_response_lens \
            --result_path $result_dir/${backend}_range_${range}_throughput
done