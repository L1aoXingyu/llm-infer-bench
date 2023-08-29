from typing import Tuple
import time
import json
import aiohttp


async def query_model_tgi(
    prompt: Tuple[str, int, int], stream: bool, port: int, num_requests: int
):
    prompt, prompt_len, expected_response_len = prompt
    assert expected_response_len > 0, f"{expected_response_len=}"

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)  # 4 hours

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": expected_response_len,
            },
        }

        start_time = time.time()
        first_chunk_time = 0
        if stream:
            async with session.post(
                f"http://localhost:{port}/generate_stream", json=generation_input
            ) as resp:
                if resp.status != 200:
                    print(f"Error: {resp.status} {resp.reason}")
                    print(await resp.text())
                    return None, None, None

                buffer = b""
                first_chunk_received = False
                async for chunk in resp.content.iter_any():
                    buffer += chunk

                    # If this is the first chunk, record the time taken
                    if not first_chunk_received:
                        first_chunk_time = time.time() - start_time
                        first_chunk_received = True

                    while b"\0" in buffer:  # Split by null character
                        json_str, buffer = buffer.split(b"\0", 1)
                output = json.loads(json_str.decode("utf-8"))  # Decode JSON
        else:
            async with session.post(
                f"http://localhost:{port}/generate", json=generation_input
            ) as resp:
                if resp.status != 200:
                    print(f"Error: {resp.status} {resp.reason}")
                    print(await resp.text())
                    return None, None, None

                output = await resp.json()

    return output["generated_text"][0], expected_response_len, first_chunk_time
