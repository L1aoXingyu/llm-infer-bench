import time
import numpy as np


class MeasureLatency:
    def __init__(self):
        self._latencies = []
        self._per_token_latencies = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)

            # Do not record latency if request failed.
            if "generated_text" in output:
                latency = time.time() - start
                self._latencies.append(latency)
                try:
                    self._per_token_latencies.append(latency / output["response_len"])
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass

            return prompt, output

        return measured


def calculate_throughput(
    queries,
    dur_s,
    backend,
    tokenizer,
    median_token_latency,
    median_e2e_latency,
    all_e2e_latencies,
    all_per_token_latencies,
    results_filename,
    log_latencies,
    fail_on_response_failure,
):
    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if "generated_text" in response:
            prompts.append(prompt)
            # [['generated_text'], ['generated_text'], ...]] -> ['generated_text', 'generated_text', ...]
            responses.append(response["generated_text"][0])
        if "naive_hf_lens" in response:
            naive_hf_lens.append(response["naive_hf_lens"])
        if "ray_gen_len" in response:
            ray_gen_lens.append(response["ray_gen_len"])
        if "num_output_tokens_cf" in response:
            cf_gen_lens.append(response["num_output_tokens_cf"])

        if "response_len" in response:
            expected_response_lens.append(response["response_len"])

    prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)["input_ids"]]
    response_ids = [r for r in tokenizer.batch_encode_plus(responses)["input_ids"]]

    print(
        f"check_len actual {list(sorted(len(response) for response in response_ids))}"
    )
    print(f"check_len expect {list(sorted(expected_response_lens))}")
    print(f"   self-reported {list(sorted(cf_gen_lens))}")

    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #    print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    print(f"naive_hf_lens {list(sorted(naive_hf_lens))}")
    print(f"prompt_lens {list(sorted(prompt_lens))}")
    print(f"calc_throughput response_lens {list(sorted(response_lens))}")
    print(f"expected_response_lens {list(sorted(expected_response_lens))}")
    print(f"ray_gen_lens {list(sorted(ray_gen_lens))}")

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum([response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum([prompt_len for prompt_len, _ in naive_hf_lens])

        response_token_count = total_prompt_tokens + total_resp_tokens

    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)

    if backend == GenerationBackend.NaiveHfPipeline:
        # It returns the prompt in the output.
        prompt_token_count = 0

    if backend == GenerationBackend.FasterTransformer:
        response_token_count = sum(expected_response_lens)

    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')

    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # print(f'throughput_tok_s {throughput_tok_s:.02f}')

    qps = len(responses) / dur_s

    with open(results_filename, "a") as f:
        msg = f"backend {backend} dur_s {dur_s:.02f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.02f} successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}, {median_token_latency=}, {median_e2e_latency=}"
        if log_latencies:
            msg += f" {all_e2e_latencies=} {all_per_token_latencies=}"
        print(msg, file=f)
        print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(
            queries
        ), f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies)
    cumsum = np.cumsum(hist)
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")
