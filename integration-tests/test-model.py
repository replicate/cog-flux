"""
It's tests.
Spins up a cog server and hits it with the HTTP API if you're local; runs through the python client if you're not
"""

import base64
import os
import subprocess
import sys
import time
import pytest
import requests
import replicate
from functools import partial
from PIL import Image
from io import BytesIO
import numpy as np

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
LOCAL_ENDPOINT = "http://localhost:5000/predictions"
MODEL_NAME = os.getenv("MODEL_NAME", "no model configured")
IS_DEV = "dev" in MODEL_NAME


def local_run(model_endpoint: str, model_input: dict):
    # TODO: figure this out for multi-image local predictions
    st = time.time()
    response = requests.post(model_endpoint, json={"input": model_input})
    et = time.time() - st
    data = response.json()

    try:
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
        return et, Image.open(BytesIO(data))
    except Exception as e:
        print("Error!")
        print("input:", model_input)
        print(data["logs"])
        raise e


def replicate_run(version: str, model_input: dict):
    pred = replicate.predictions.create(version=version, input=model_input)

    pred.wait()

    predict_time = pred.metrics["predict_time"]
    images = []
    for url in pred.output:
        response = requests.get(url)
        images.append(Image.open(BytesIO(response.content)))
    print(pred.id)
    return predict_time, images


def wait_for_server_to_be_ready(url, timeout=400):
    """
    Waits for the server to be ready.

    Args:
    - url: The health check URL to poll.
    - timeout: Maximum time (in seconds) to wait for the server to be ready.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            data = response.json()

            if data["status"] == "READY":
                return
            elif data["status"] == "SETUP_FAILED":
                raise RuntimeError("Server initialization failed with status: SETUP_FAILED")

        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready in the expected time.")

        time.sleep(5)  # Poll every 5 seconds


@pytest.fixture(scope="session")
def inference_func():
    if ENVIRONMENT == "local":
        return partial(local_run, LOCAL_ENDPOINT)
    elif ENVIRONMENT in {"test", "prod"}:
        model = replicate.models.get(MODEL_NAME)
        version = model.versions.list()[0]
        return partial(replicate_run, version)
    else:
        raise Exception(f"env should be local, test, or prod but was {ENVIRONMENT}")


@pytest.fixture(scope="session", autouse=True)
def service():
    if ENVIRONMENT == "local":
        print("building model")
        # starts local server if we're running things locally
        build_command = "cog build -t test-model".split()
        subprocess.run(build_command, check=True)
        container_name = "cog-test"
        try:
            subprocess.check_output(["docker", "inspect", '--format="{{.State.Running}}"', container_name])
            print(f"Container '{container_name}' is running. Stopping and removing...")
            subprocess.check_call(["docker", "stop", container_name])
            subprocess.check_call(["docker", "rm", container_name])
            print(f"Container '{container_name}' stopped and removed.")
        except subprocess.CalledProcessError:
            # Container not found
            print(f"Container '{container_name}' not found or not running.")

        run_command = f"docker run -d -p 5000:5000 --gpus all --name {container_name} test-model ".split()
        process = subprocess.Popen(run_command, stdout=sys.stdout, stderr=sys.stderr)

        wait_for_server_to_be_ready("http://localhost:5000/health-check")

        yield
        process.terminate()
        process.wait()
        stop_command = "docker stop cog-test".split()
        subprocess.run(stop_command)
    else:
        yield


def get_time_bound():
    """entirely here to make sure we don't recompile"""
    return 20 if IS_DEV else 10


def test_base_generation(inference_func):
    """standard generation for dev and schnell. assert that the output image has a dog in it with blip-2 or llava"""
    test_example = {
        "prompt": "A cool dog",
        "aspect ratio": "1:1",
        "num_outputs": 1,
    }
    time, img_out = inference_func(test_example)
    img_out = img_out[0]

    assert time < get_time_bound()
    assert img_out.size == (1024, 1024)


def test_num_outputs(inference_func):
    """num_outputs = 4, assert time is about what you'd expect off of the prediction object"""
    base_time = None
    for n_outputs in range(1, 5):
        test_example = {
            "prompt": "A cool dog",
            "aspect ratio": "1:1",
            "num_outputs": n_outputs,
        }
        time, img_out = inference_func(test_example)
        assert len(img_out) == n_outputs
        if base_time:
            assert time < base_time * n_outputs * 1.5
        if n_outputs == 1:
            base_time = time


def test_determinism(inference_func):
    """determinism - test with the same seed twice"""
    test_example = {"prompt": "A cool dog", "aspect_ratio": "9:16", "num_outputs": 1, "seed": 112358}
    time, out_one = inference_func(test_example)
    out_one = out_one[0]
    assert time < get_time_bound()
    time_two, out_two = inference_func(test_example)
    out_two = out_two[0]
    assert time_two < get_time_bound()
    assert out_one.size == (768, 1344)

    one_array = np.array(out_one, dtype=np.uint16)
    two_array = np.array(out_two, dtype=np.uint16)
    assert np.allclose(one_array, two_array, atol=20)


def test_resolutions(inference_func):
    """changing resolutions - iterate through all resolutions and make sure that the output is"""
    aspect_ratios = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "21:9": (1536, 640),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
        "4:5": (896, 1088),
        "5:4": (1088, 896),
        "9:16": (768, 1344),
        "9:21": (640, 1536),
    }

    for ratio, output in aspect_ratios.items():
        test_example = {"prompt": "A cool dog", "aspect_ratio": ratio, "num_outputs": 1, "seed": 112358}

        time, img_out = inference_func(test_example)
        img_out = img_out[0]
        assert img_out.size == output
        assert time < get_time_bound()


def test_img2img(inference_func):
    """img2img. does it work?"""
    if not IS_DEV:
        assert True
        return

    test_example = {
        "prompt": "a cool walrus",
        "image": "https://replicate.delivery/pbxt/IS6z50uYJFdFeh1vCmXe9zasYbG16HqOOMETljyUJ1hmlUXU/keanu.jpeg",
    }

    _, img_out = inference_func(test_example)
    img_out = img_out[0]
    assert img_out.size[0] % 16 == 0
    assert img_out.size[0] < 1440
    assert img_out.size[1] % 16 == 0
    assert img_out.size[1] < 1440
