#!/usr/bin/env bash

export PORT=18082

backend="lightllm"
model_name="/data/Llama-2-13b-hf"
result_dir="./benchmark_llama13b"
# model_name="/data/Llama-2-70b-chat-hf"
# result_dir="./benchmark_llama70b"

# Run real test
for qps in 2 4 8; do
    QPS=$qps
    range=512
    num_prompts=1000
    echo "range $range, num_prompts $num_prompts, qps $QPS"

    ulimit -n 65536 && python3 benchmark_llm_serving.py \
            --backend $backend \
            --port $PORT \
            --random_prompt_lens_mean 256 \
            --random_prompt_lens_range 256 \
            --num_requests $num_prompts \
            --stream True \
            --traffic_distribution poisson \
            --allow_random_response_lens \
            --random_response_lens_mean 128 \
            --random_response_lens_range $range \
            --response_distribution capped_exponential \
            --allow_random_response_lens \
            --qps $QPS \
            --result_path $result_dir/${backend}_qps_${qps}_latency
done