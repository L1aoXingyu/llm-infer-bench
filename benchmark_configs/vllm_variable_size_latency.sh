#!/usr/bin/env bash

export PORT=8000

model_name="/data/dataset/Llama-2-7b-hf"
cuda_devices="7"
result_dir="./benchmark_llama7b_new"
backend="vllm"

function start_model_server {

    ulimit -n 65536 && CUDA_VISIBLE_DEVICES=$cuda_devices python3 -m vllm.entrypoints.api_server \
        --port $PORT \
        --model $model_name \
        --use-np-weights \
        --max-num-batched-tokens 8700 \
        2>&1 &

    # Wait for the server to be ready
    until curl -s http://localhost:${PORT} > /dev/null; do
        echo "Waiting for model server to start..."
        sleep 1
    done

    echo "model server started on port $PORT"
}

function kill_model_server {
    echo 'killing model server'
    ps aux | grep 'vllm.entrypoints.api_server' | grep -v 'vim' | awk '{print $2}' | xargs kill -9
    wait
}

trap kill_model_server EXIT

# Catch OOMs early.
start_model_server

python3 benchmark_llm_serving.py \
            --backend $backend \
            --port $PORT \
            --random_prompt_lens_mean 512 \
            --random_prompt_lens_range 0 \
            --stream True \
            --num_requests 30 \
            --allow_random_response_lens \
            --fixed_max_response_tokens 1536
popd
kill_model_server

# Run real test
for qps in 1 2 4; do
    QPS=$qps
    range=512
    num_prompts=1000
    echo "range $range, num_prompts $num_prompts, qps $QPS"

    start_model_server

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
    popd
    kill_model_server

done