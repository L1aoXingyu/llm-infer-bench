from typing import Tuple
import aiohttp
import os


async def query_model_hf(
    prompt: Tuple[str, int, int], stream: bool, port: int, num_requests: int
):
    prompt, prompt_len, expected_response_len = prompt

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    bs = int(os.environ.get("NAIVE_HF_BS", 1))

    first_chunk_time = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "total_benchmark_requests": num_requests,
            "inputs": prompt,
            "parameters": {
                "batch_size": bs,
                "max_length": expected_response_len + prompt_len,
                "prompt_len": prompt_len,
                "response_len": expected_response_len,
                "allow_variable_generation_length": True,
            },
        }

        async with session.post(
            f"http://localhost:{port}/generate", json=generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None, None

            output = await resp.json()

    return output["generated_text"], expected_response_len, first_chunk_time
