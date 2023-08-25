#!/usr/bin/env bash

export PORT=8000

num_prompts=1000
ranges=("32" "128" "512" "1536")
max_batch_total_tokens_vals=("8700" "8700" "8700" "8700")
model_name="/data/dataset/Llama-2-7b-hf"
cuda_devices="7"
result_dir="./benchmark_llama7b"
backend="lightllm"

function start_model_server {
    local max_num_batched_tokens=$1

    ulimit -n 65536 && CUDA_VISIBLE_DEVICES=$cuda_devices python3 -m vllm.entrypoints.api_server \
        --port $PORT \
        --model $model_name \
        --use-np-weights \
        --max-num-batched-tokens $max_num_batched_tokens \
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
for i in ${!ranges[@]}; do
    range=${ranges[$i]}
    max_num_batched_tokens=${max_batch_total_tokens_vals[$i]}
    echo "range $range, max_num_batched_tokens $max_num_batched_tokens"

    start_model_server $max_num_batched_tokens

    python3 benchmark_llm_serving.py \
            --backend $backend \
            --port $PORT \
            --random_prompt_lens_mean 512 \
            --random_prompt_lens_range 0 \
            --num_requests 30 \
            --allow_random_response_lens \
            --fixed_max_response_tokens $range \
    popd
    kill_model_server

done

# Run real test
for i in ${!ranges[@]}; do
    range=${ranges[$i]}
    max_num_batched_tokens=${max_batch_total_tokens_vals[$i]}
    echo "range $range, max_num_batched_tokens $max_num_batched_tokens"

    start_model_server $max_num_batched_tokens

    ulimit -n 65536 && python3 benchmark_llm_serving.py \
            --backend $backend \
            --port $PORT \
            --random_prompt_lens_mean 512 \
            --random_prompt_lens_range 0 \
            --num_requests $num_prompts \
            --allow_random_response_lens \
            --random_response_lens_mean 128 \
            --random_response_lens_range $range \
            --response_distribution capped_exponential \
            --allow_random_response_lens \
            --result_path $result_dir/${backend}_range_${range}
    popd
    kill_model_server

done