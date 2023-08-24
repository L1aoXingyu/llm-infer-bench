import time
import numpy as np
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase


def get_token_id_lens(tokenizer, batch):
    return [len(s) for s in tokenizer(batch)["input_ids"]]


class MeasureLatency:
    def __init__(self):
        self._latencies = []
        self._per_token_latencies = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            generated_text, response_len = await f(*args, **kwargs)

            # Do not record latency if request failed.
            if generated_text:
                latency = time.time() - start
                self._latencies.append(latency)
                try:
                    self._per_token_latencies.append(latency / response_len)
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass

            return generated_text

        return measured


def calculate_throughput(
    prompts: List[Tuple[str, int, int]],
    responses: List[str],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    median_token_latency: float,
    median_e2e_latency: float,
    results_filename: str,
    fail_on_response_failure: bool,
):
    prompt_lens = []
    expected_response_lens = []
    for p in prompts:
        prompt_lens.append(p[1])
        expected_response_lens.append(p[2])

    response_lens = get_token_id_lens(tokenizer, responses)
    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)
    expected_response_token_count = sum(expected_response_lens)

    # There are some difference between the token count of response and expected_response
    # this is because tokenizer is not 1:1 map
    print(
        f"{prompt_token_count=}, {response_token_count=}, {expected_response_token_count=}"
    )

    throughput_token_s = (prompt_token_count + expected_response_token_count) / dur_s

    qps = len(responses) / dur_s

    with open(results_filename, "a") as f:
        msg = f"dur_s {dur_s:.02f} tokens_per_s {throughput_token_s:.02f} qps {qps:.02f} successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}, {median_token_latency=}, {median_e2e_latency=}"
        print(msg, file=f)
        print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(
            prompt_lens
        ), f"{fail_on_response_failure=}, expected number of successful responses to equal number of queries, got {len(responses)} vs {len(prompt_lens)}"


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies)
    cumsum = np.cumsum(hist)
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")
