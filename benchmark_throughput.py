from dataclasses import dataclass, field
import time
import random
import numpy as np
import aiohttp
from utils import MeasureLatency, calculate_throughput, calculate_cdf
import itertools
import json
from typing import Optional, Union, List, Tuple
from enum import Enum
import asyncio
from transformers import HfArgumentParser, PreTrainedTokenizerBase, AutoTokenizer


class Distribution(str, Enum):
    burst = "burst"
    uniform = "uniform"
    exponential = "exponential"
    capped_exponential = "capped_exponential"


@dataclass
class InferenceArguments:
    port: Optional[int] = field(default=8000, metadata={"help": "Port to listen on"})
    prompt_filename: Optional[str] = field(
        default=None, metadata={"help": "Path to the benchmarking dataset filename"}
    )
    num_requests: Optional[int] = field(
        default=100,
        metadata={"help": "Number of requests to profile inference benchmark"},
    )
    random_prompt_lens_mean: Optional[int] = field(
        default=128, metadata={"help": "Mean of the random prompt lengths"}
    )
    random_prompt_lens_range: Optional[int] = field(
        default=20, metadata={"help": "Range of the random prompt lengths"}
    )
    allow_random_response_lens: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Allow random response lengths, otherwise use fixed_max_response_tokens"
        },
    )
    fixed_max_response_tokens: Optional[int] = field(
        default=1024, metadata={"help": "Fixed max tokens for response length"}
    )
    max_model_input_length: Optional[int] = field(
        default=2048, metadata={"help": "Max model input length"}
    )
    random_response_lens_mean: Optional[int] = field(
        default=128, metadata={"help": "Mean of the random response lengths"}
    )
    random_response_lens_range: Optional[int] = field(
        default=20, metadata={"help": "Range of the random response lengths"}
    )
    tokenizer: Optional[str] = field(
        default="facebook/opt-125m", metadata={"help": "Tokenizer to use"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Trust remote code"}
    )
    traffic_distribution: Optional[Distribution] = field(
        default=Distribution.burst,
        metadata={"help": "Distribution of the requests sending"},
    )
    response_distribution: Optional[Distribution] = field(
        default=Distribution.uniform,
        metadata={"help": "Distribution of the response lengths"},
    )
    print_generation_lens_and_exit: Optional[bool] = field(
        default=False, metadata={"help": "Print generation lens and exit"}
    )


def sample_requests(
    tokenizer: PreTrainedTokenizerBase,
    prompt_lens_mean,
    prompt_lens_range,
    num_prompts,
    prompt_filename: str,
) -> Tuple[List[str], List[int]]:
    if prompt_filename:
        # Load real dataset
        with open(prompt_filename, "r") as f:
            prompts = json.load(f)
        prompt_lens = itertools.repeat(-1)
    else:
        # Generate random prompts
        random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts(
            tokenizer, prompt_lens_mean, prompt_lens_range, num_prompts
        )

    return prompts, prompt_lens


def gen_random_prompts(
    tokenizer: PreTrainedTokenizerBase,
    lens_mean: int,
    lens_range: int,
    num_prompts: int,
):
    low = lens_mean - (lens_range // 2)
    high = lens_mean + (lens_range // 2)
    max_vocab_ids = max(tokenizer.get_vocab().values())

    def gen_prompt_tokens(length):
        return [random.randint(10, max_vocab_ids) for _ in range(length)]

    prompt_lens = list(map(lambda _: random.randint(low, high), range(num_prompts)))
    prompts_as_tokens = list(
        map(lambda prompt_len: gen_prompt_tokens(prompt_len), prompt_lens)
    )
    prompts = list(
        map(lambda prompt_tokens: tokenizer.decode(prompt_tokens), prompts_as_tokens)
    )

    # Because tokens do not map 1:1 to words, sometimes we get more or less tokens than desired.
    new_prompts = []
    encoded_prompts = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    for encoded, l in zip(encoded_prompts, prompt_lens):
        if len(encoded) > l:
            # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
            encoded = encoded[:l]
        elif len(encoded) < l:
            # This left-pads the prompt with padding tokens.
            encoded = [tokenizer.pad_token_id] * (l - len(encoded)) + encoded
        decoded = tokenizer.decode(encoded)
        encoded = tokenizer(decoded, add_special_tokens=False)["input_ids"]
        assert (
            len(encoded) == l
        ), f"Expected prompt to contain exactly {l} tokens, got {len(encoded)=}"
        new_prompts.append(decoded)

    return new_prompts, prompt_lens


def gen_random_response_lens(
    distribution: Distribution, len_mean: int, len_range: int, num_responses: int
):
    if distribution == Distribution.uniform:
        if len_range == 0:
            return [len_mean for _ in range(num_responses)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        num_to_generate = list(
            map(lambda _: random.randint(low, high), range(num_responses))
        )
        return num_to_generate
    elif distribution == Distribution.exponential:
        np.random.seed(random.randint(0, 1e6))
        return [
            min(round(s), len_range)
            for s in np.random.exponential(scale=len_mean, size=num_responses)
        ]
    elif distribution == Distribution.capped_exponential:
        np.random.seed(random.randint(0, 1e6))
        response_lens = []
        while len(response_lens) < num_responses:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range:
                response_lens.append(sample)
        return response_lens
    else:
        raise ValueError(f"unknown distribution {distribution=}")


async def query_model_vllm(prompt: Tuple[str, int, int], port: int):
    prompt, _, expected_response_len = prompt

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)  # 4 hours

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = dict(
            inputs=prompt,
            parameters=dict(ignore_eos=True, max_tokens=expected_response_len),
        )

        async with session.post(
            f"http://localhost:{port}/generate", json=generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None

            output = await resp.json()

            return output["generated_text"][0], expected_response_len


async def async_request_gen(generator, qps: float, distribution: Distribution):
    def get_wait_time():
        mean_time_between_requests = 1.0 / qps
        if distribution == Distribution.uniform:
            return mean_time_between_requests
        else:
            return np.random.exponential(mean_time_between_requests)

    while True:
        try:
            item = next(generator)
            yield item
            if distribution != Distribution.burst:
                await asyncio.sleep(get_wait_time())
        except StopIteration:
            return


async def benchmark(
    prompts: List[Tuple[str, int, int]],
    tokenizer: PreTrainedTokenizerBase,
    traffic_distribution: Distribution,
    port: int,
):
    m = MeasureLatency()
    query_model = m.measure(query_model_vllm)

    if traffic_distribution == Distribution.burst:
        qps = float("inf")

    print(f"Starting benchmark with {traffic_distribution=}, {qps=}")

    total_requests = len(prompts[0])

    async_prompts = async_request_gen(iter(prompts), qps, traffic_distribution)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(prompt, port)))

    responses = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time

    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._latencies)

    calculate_throughput(
        prompts,
        responses,
        dur_s,
        tokenizer,
        median_token_latency,
        median_e2e_latency,
        "results.json",
        True,
    )
    # calculate_cdf()


def main():
    parser = HfArgumentParser((InferenceArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    prompts, prompt_lens = sample_requests(
        tokenizer,
        args.random_prompt_lens_mean,
        args.random_prompt_lens_range,
        args.num_requests,
        args.prompt_filename,
    )

    if args.allow_random_response_lens:
        response_lens = gen_random_response_lens(
            args.response_distribution,
            args.random_response_lens_mean,
            args.random_response_lens_range,
            num_responses=args.num_requests,
        )
    else:
        response_lens = [
            args.fixed_max_response_tokens for _ in range(args.num_requests)
        ]

    for i, (prompt_len, resp_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + resp_len
        if total > args.max_model_input_length:
            print(f"truncating long prompt+resp_len {prompt_len=} {resp_len=}")
            resp_len = args.max_model_input_length - prompt_len
        response_lens[i] = resp_len

    print(f"{prompt_lens[:5]=}")
    print(f"{response_lens[:5]=}")
    if args.print_generation_lens_and_exit:
        return

    prompts = list(zip(prompts, prompt_lens, response_lens))

    asyncio.run(benchmark(prompts, tokenizer, args.traffic_distribution, args.port))


if __name__ == "__main__":
    main()
