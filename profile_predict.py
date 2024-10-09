import os
import shlex
import shutil
import subprocess
import pathlib

import cog
import requests
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def predict(
        self,
        nsys_args: str = Input(default="--python-sampling=true"),
        predict_args: dict = Input(default={"prompt": "hi"}),
        n_runs: int = 3,
        # model: str = "SchnellPredictor",
    ):
        shutil.copyfile("/src/inner.cog.yaml", "/src/cog.yaml")
        command = shlex.shlex(f"nsys profile {nsys_args} python3 -m cog.server.http")
        proc = subprocess.Popen(command, env={"PORT": "5001"})
        for i in range(n_runs):
            requests.post("localhost:5001/predictions", json=predict_args).json()
        proc.send_signal(signal.SIGINT)
        proc.wait()
        report = sorted(pathlib.Path(".").glob("report*"), key=os.path.getmtime)[-1]
        return cog.Path(report)
