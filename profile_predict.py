import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import time

import cog
import requests
from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def predict(
        self,
        nsys_args: str = Input(default="--python-sampling=true"),
        # why don't we allow dict inputs again?
        predict_args: str = Input(default='{"prompt":"hi"}'),
        n_runs: int = 3,
        # model: str = "SchnellPredictor",
    ) -> cog.Path:
        shutil.copyfile("/src/inner.cog.yaml", "/src/cog.yaml")
        python = shutil.which("python3")
        command = shlex.split(
            f"/usr/local/bin/nsys profile {nsys_args} {python} -m cog.server.http"
        )
        print("running", command)
        proc = subprocess.Popen(command, env={**os.environ, "PORT": "5001"})
        for i in range(10 * 60 // 5):  # 10m
            try:
                check = requests.get("http://localhost:5001/health-check").json()
                if check.get("status") == "READY":
                    print("cog is ready", check)
                    break
            except (json.JSONDecodeError, requests.RequestException) as e:
                if i > 12:
                    raise Exception("cog didn't come up after a minute") from e
            time.sleep(5)
        for i in range(n_runs):  # not quite right
            data = requests.post("http://localhost:5001/predictions", json={"input":json.loads(predict_args)}).json()
            if isinstance(data, dict):
                if data.get("logs"):
                    data["logs"] = data["logs"][:1024]
                if data.get("output"):
                    data["output"] = data["output"][:128]
            print(data)
        proc.send_signal(signal.SIGINT)
        proc.wait()
        report = sorted(pathlib.Path(".").glob("report*"), key=os.path.getmtime)[-1]
        return cog.Path(report)
