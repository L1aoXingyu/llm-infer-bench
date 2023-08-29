#!/usr/bin/env python3

import torch
import uvicorn
from fastapi import FastAPI, Request
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
import argparse
import asyncio
from typing import Dict
import os

os.environ["TRANSFORMERS_CACHE"] = "/data/cache"

assert (
    "CUDA_VISIBLE_DEVICES" in os.environ
), "Set CUDA_VISIBLE_DEVICES, else this will take memory on each (and load model to 0)"


app = FastAPI()


class FastAPIServer:
    def __init__(self, model_name):
        assert model_name is not None
        self.model_name = model_name
        self.model = None
        self.pipe = None
        self.tokenizer = None
        self.init_model()

        self.waiting_requests = []
        self.responses = []
        self.end_event = None
        self.waiting_expectation = None

    def init_model(self):
        print("Init model")
        self._inference_mode_raii_guard = torch._C._InferenceMode(True)

        config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = config.eos_token_id

        dtype = torch.float16
        device = "cuda:0"
        with deepspeed.OnDevice(dtype=dtype, device=device):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.float16,
                # device='cuda:0',
            )

        self.model = model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=model,
            config=config,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype=dtype,
            return_full_text=False,
        )

        print("Model ready")

    async def _wait_until_all_requests(
        self, waiting_expectation, prompt, sampling_config
    ):
        if self.end_event is None:
            self.end_event = asyncio.Event()
        rank = len(self.waiting_requests)
        self.waiting_requests.append((prompt, sampling_config))
        if len(self.waiting_requests) == waiting_expectation:
            return True, rank
        else:
            await self.end_event.wait()
            return False, rank

    async def generate(self, request_dict: Dict):
        total_benchmark_requests = request_dict.get("total_benchmark_requests", None)
        if total_benchmark_requests is None:
            raise ValueError(
                f"total_benchmark_requests invalid {total_benchmark_requests}"
            )

        if self.waiting_expectation is None:
            self.waiting_expectation = total_benchmark_requests
            print(f"setting waiting expectation to {total_benchmark_requests=}")
        elif self.waiting_expectation != total_benchmark_requests:
            raise ValueError(
                f"waiting inconsistency {self.waiting_expectation} {total_benchmark_requests}"
            )

        prompt = request_dict["inputs"]
        sampling_config = request_dict["parameters"]

        # print(f'generate, prompt "{prompt}"')

        should_call_forward, rank = await self._wait_until_all_requests(
            total_benchmark_requests,
            prompt,
            sampling_config,
        )

        if should_call_forward:
            print("submitting forward pass")
            try:
                self.responses = self._forward_pass(self.waiting_requests)
            except torch.cuda.OutOfMemoryError as e:
                self.responses = [
                    [{"generated_text": None, "error": e}]
                    for _ in range(self.waiting_expectation)
                ]
            print("forward pass done")

            self.end_event.set()
            self.waiting_requests = []
            self.end_event.clear()  # race here
            self.waiting_expectation = None

        result = {
            "generated_text": self.responses[rank][0]["generated_text"],
        }
        assert (
            "error" not in self.responses[rank][0]
        ), f"got error {self.responses[rank][0]}"
        return result

    def _forward_pass(self, all_requests):
        prompts = []

        batch_size = None
        max_length = None
        allow_var_len = None
        prompt_len = None
        for prompt, parameters in all_requests:
            if max_length is None:
                max_length = parameters["max_length"]
            else:
                max_length = max(max_length, parameters["max_length"])
                # assert max_length == parameters['max_length'], f"incorrect max_length {max_length} {parameters['max_length']}"

            if batch_size is None:
                batch_size = parameters["batch_size"]
            else:
                assert (
                    batch_size == parameters["batch_size"]
                ), f"incorrect batch_size {batch_size} {parameters['batch_size']}"

            if allow_var_len is None:
                allow_var_len = parameters["allow_variable_generation_length"]
            else:
                assert (
                    allow_var_len == parameters["allow_variable_generation_length"]
                ), f"incorrect allow_var_len {allow_var_len} {parameters['allow_variable_generation_length']}"

            if prompt_len is None:
                prompt_len = parameters["prompt_len"]
            else:
                prompt_len = max(prompt_len, parameters["prompt_len"])
                # assert prompt_len == parameters['prompt_len'], f"incorrect prompt_len {prompt_len} {parameters['prompt_len']}"

            prompts.append(prompt)

        if allow_var_len and False:
            min_length = 0
        else:
            min_length = max_length

        def data_gen():
            for i, prompt in enumerate(prompts):
                encoded = self.tokenizer(prompt)
                l = len(encoded["input_ids"])
                print(f"{i=} prompt tok len {l}")

                yield prompt

        print(
            f"max_prompt_len {prompt_len} max_length {max_length} min_length {min_length} allow_var_len {allow_var_len}"
        )

        outputs = []
        for out in self.pipe(
            data_gen(),
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            batch_size=batch_size,
        ):
            outputs.append(out)

        return outputs


@app.post("/generate")
async def generate_stream(request: Request):
    request_dict = await request.json()
    return await server.generate(request_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    server = FastAPIServer(args.model_name)

    uvicorn.run(app, host="localhost", port=args.port, log_level="info")
