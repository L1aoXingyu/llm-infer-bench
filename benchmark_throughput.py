from dataclasses import dataclass, field
import random
import numpy as np
import itertools
import json
from typing import Optional, Union
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
        default=1000,
        metadata={"help": "Number of requests to profile inference benchmark"},
    )
    random_prompt_lens_mean: Optional[int] = field(
        default=128, metadata={"help": "Mean of the random prompt lengths"}
    )
    random_prompt_lens_range: Optional[int] = field(
        default=20, metadata={"help": "Range of the random prompt lengths"}
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
    response_distribution: Optional[Distribution] = field(
        default=Distribution.uniform,
        metadata={"help": "Distribution of the response lengths"},
    )


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
    # Confusingly, it works with a single iteration per prompt.
    for i, (p, l) in enumerate(zip(prompts, prompt_lens)):
        encoded = tokenizer(p, add_special_tokens=False)["input_ids"]
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
        prompts[i] = decoded

    return prompts, prompt_lens


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


async def benchmark():
    pass


def main():
    parser = HfArgumentParser((InferenceArguments,))
    (inference_args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        inference_args.tokenizer, trust_remote_code=inference_args.trust_remote_code
    )

    if inference_args.prompt_filename:
        # Load real dataset
        with open(inference_args.prompt_filename, "r") as f:
            prompts = json.load(f)
        prompt_lens = itertools.repeat(-1)
    else:
        # Generate random prompts
        random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts(
            tokenizer,
            inference_args.lens_mean,
            inference_args.lens_range,
            inference_args.num_requests,
        )

        response_lens = gen_random_response_lens(
            inference_args.response_distribution,
            inference_args.random_response_lens_mean,
            inference_args.random_response_lens_range,
            num_responses=inference_args.num_requests,
        )

    # asyncio.run(benchmark())


if __name__ == "__main__":
    main()
