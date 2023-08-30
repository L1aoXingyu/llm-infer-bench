from typing import Tuple
import aiohttp
import time
import json


async def query_model_lightllm(
    prompt: Tuple[str, int, int], stream: bool, port: int, num_requests: int
):
    prompt, _, expected_response_len = prompt
    assert expected_response_len > 0, f"{expected_response_len=}"
    assert stream, "lightllm only supports streaming with True"

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)  # 4 hours

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": expected_response_len,
            },
        }

        start_time = time.time()
        first_chunk_time = 0
        async with session.post(
            f"http://localhost:{port}/generate", json=generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None, None

            if stream:
                first_chunk_received = False
                chunks = []
                async for chunk, _ in resp.content.iter_chunks():
                    # If this is the first chunk, record the time taken
                    if not first_chunk_received:
                        first_chunk_time = time.time() - start_time
                        first_chunk_received = True

                    chunks.append(chunk)

                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)
            else:
                raise NotImplementedError

            return output["generated_text"][0], expected_response_len, first_chunk_time
