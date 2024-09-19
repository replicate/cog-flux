"""
A handy utility for verifying image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""

import base64
import sys
import time
from pathlib import Path
import requests


def gen(output_fn, **kwargs):
    st = time.time()
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()
    print("Generated in: ", time.time() - st)

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except Exception:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    Path(output_fn).write_bytes(data)


def test_fp8_and_bf16():
    """
    runs generations in fp8 and bf16 on the same node! wow!
    """
    gen(
        "float8_dog.png",
        prompt="a cool dog",
        aspect_ratio="1:1",
        num_outputs=1,
        output_format="png",
        disable_safety_checker=True,
        seed=123,
        float_8=True,
    )

    gen(
        "bf16_dog.png",
        prompt="a cool dog",
        aspect_ratio="1:1",
        num_outputs=1,
        output_format="png",
        disable_safety_checker=True,
        seed=123,
        float_8=False,
    )

    gen(
        "float8_dog_2.png",
        prompt="a cool dog",
        aspect_ratio="2:3",
        num_outputs=1,
        output_format="png",
        disable_safety_checker=True,
        seed=1231,
        float_8=True,
    )

    gen(
        "bf16_dog_2.png",
        prompt="a cool dog",
        aspect_ratio="2:3",
        num_outputs=1,
        output_format="png",
        disable_safety_checker=True,
        seed=1231,
        float_8=False,
    )


if __name__ == "__main__":
    test_fp8_and_bf16()
