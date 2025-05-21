import base64
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from contextlib import contextmanager

from cog import Secret
from huggingface_hub import HfApi, hf_hub_download, login, logout

DEFAULT_CACHE_BASE_DIR = Path("/src/weights-cache")

from dotenv import load_dotenv

load_dotenv()


class WeightsDownloadCache:
    def __init__(
        self, min_disk_free: int = 10 * (2**30), base_dir: Path = DEFAULT_CACHE_BASE_DIR
    ):
        self.min_disk_free = min_disk_free
        self.base_dir = base_dir
        self.hits = 0
        self.misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        base_dir.mkdir(parents=True, exist_ok=True)

    def ensure(
        self,
        url: str,
        hf_api_token: Secret | None = None,
        civitai_api_token: Secret | None = None,
    ) -> Path:
        path = self._weights_path(url)

        if path in self.lru_paths:
            # here we remove to re-add to the end of the LRU (marking it as recently used)
            self.hits += 1
            self.lru_paths.remove(path)
        elif not Path.exists(
            path
        ):  # local dev; sometimes we'll have a lora already downloaded
            self.misses += 1

            while not self._has_enough_space() and len(self.lru_paths) > 0:
                self._remove_least_recent()

            download_weights(
                url,
                path,
                hf_api_token=hf_api_token,
                civitai_api_token=civitai_api_token,
            )

        self.lru_paths.append(path)  # Add file to end of cache
        return path

    def cache_info(self) -> str:
        return f"CacheInfo(hits={self.hits}, misses={self.misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _remove_least_recent(self) -> None:
        oldest = self.lru_paths.popleft()
        print("removing oldest", oldest)
        oldest.unlink()

    def _has_enough_space(self) -> bool:
        disk_usage = shutil.disk_usage(self.base_dir)

        free = disk_usage.free
        print(f"{free=}")  # TODO(andreas): remove debug

        return free >= self.min_disk_free

    def _weights_path(self, url: str) -> Path:
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]  # Use the first 16 characters of the hash
        return self.base_dir / short_hash


def download_weights(
    url: str,
    path: Path,
    hf_api_token: str | None = None,
    civitai_api_token: str | None = None,
):
    download_url = make_download_url(url, civitai_api_token=civitai_api_token)
    download_weights_url(download_url, path, hf_api_token=hf_api_token)


@contextmanager
def logged_in_to_huggingface(
    token: Secret | None = None, add_to_git_credential: bool = False
):
    """Context manager for temporary Hugging Face login."""
    try:
        if token is not None:
            print("Attemptig to login to HuggingFace using provided token...")
            # Log in at the start of the context
            login(
                token=token.get_secret_value(),
                add_to_git_credential=add_to_git_credential,
            )
            print("Login to HuggingFace successful!")
        yield
    finally:
        # Always log out at the end, even if an exception occurs
        logout()
        print("Logged out of HuggingFace.")


def download_weights_url(url: str, path: Path, hf_api_token: str | None = None):
    path = Path(path)

    print("Downloading weights")
    start_time = time.time()

    if m := re.match(
        r"^(?:https?://)?huggingface\.co/([^/]+)/([^/]+)(?:/([^/]+\.safetensors))?/?$",
        url,
    ):
        if len(m.groups()) == 2:
            owner, model_name = m.groups()
            lora_weights = None
        else:
            owner, model_name, lora_weights = m.groups()

        # Use HuggingFace Hub download directly
        try:
            with logged_in_to_huggingface(hf_api_token):
                if lora_weights is None:
                    api = HfApi()
                    files = api.list_repo_files(repo_id)
                    sft_files = [file for file in files if ".safetensors" in file]
                    if len(sft_files) == 1:
                        hf_hub_download(repo_id=repo_id, filename=sft_files[0])
                    else:
                        raise ValueError(
                            f"No *.safetensors file was explicitly specified from the HuggingFace repo {repo_id} and more than one *.safetensors file was found. Found: {[stf_file for sft_file in sft_files]}"
                        )

                safetensors_path = hf_hub_download(
                    repo_id=f"{owner}/{model_name}",
                    filename=lora_weights,
                    # token=hf_api_token.get_secret_value() if hf_api_token is not None else None,
                )
                # Copy the downloaded file to the desired path
                shutil.copy(Path(safetensors_path), path)
                print(f"Downloaded {lora_weights} from HuggingFace to {path}")
        except Exception as e:
            raise ValueError(f"Failed to download from HuggingFace: {e}")
    elif url.startswith("data:"):
        download_data_url(url, path)
    elif url.endswith(".tar"):
        download_safetensors_tarball(url, path)
    elif (
        url.endswith(".safetensors")
        or "://civitai.com/api/download" in url
        or ".safetensors?" in url
    ):
        download_safetensors(url, path)
    elif url.endswith("/_weights"):
        download_safetensors_tarball(url, path)
    else:
        raise ValueError("URL must end with either .tar or .safetensors")

    print(f"Downloaded weights in {time.time() - start_time:.2f}s")


def find_safetensors(directory: Path) -> list[Path]:
    safetensors_paths = []
    for root, _, files in os.walk(directory):
        root = Path(root)
        for filename in files:
            path = root / filename
            if path.suffix == ".safetensors":
                safetensors_paths.append(path)
    return safetensors_paths


def download_safetensors_tarball(url: str, path: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_dir = temp_dir / "weights"

        try:
            subprocess.run(
                ["pget", "--log-level=WARNING", "-x", url, extract_dir], check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download tarball: {e}")

        safetensors_paths = find_safetensors(extract_dir)
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in tarball")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in tarball")
        safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_data_url(url: str, path: Path):
    _, encoded = url.split(",", 1)
    data = base64.b64decode(encoded)

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
            tar.extractall(path=temp_dir)

        safetensors_paths = find_safetensors(Path(temp_dir))
        if not safetensors_paths:
            raise ValueError("No .safetensors file found in data URI")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in data URI")
        safetensors_path = safetensors_paths[0]

        shutil.move(safetensors_path, path)


def download_safetensors(url: str, path: Path):
    try:
        # don't want to leak civitai api key
        output_redirect = subprocess.PIPE
        if "token=" in url:
            # print url without token
            print(f"downloading weights from {url.split('token=')[0]}token=***")
        else:
            print(f"downloading weights from {url}")

        result = subprocess.run(
            ["pget", url, str(path)],
            check=False,
            stdout=output_redirect,
            stderr=output_redirect,
            text=True,
        )

        if result.returncode != 0:
            error_output = result.stderr or ""
            if "401" in error_output:
                raise RuntimeError(
                    "Authorization to download weights failed. Please check to see if an API key is needed and if so pass in with the URL."
                )
            if "404" in error_output:
                if "civitai" in url:
                    raise RuntimeError(
                        "Model not found on CivitAI at that URL. Double check the CivitAI model ID; the id on the download link can be different than the id to browse to the model page."
                    )
                raise RuntimeError(
                    "Weights not found at the supplied URL. Please check the URL."
                )
            raise RuntimeError(f"Failed to download safetensors file: {error_output}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download safetensors file: {e}")


def make_download_url(url: str, civitai_api_token: str | None = None) -> str:
    if url.startswith("data:"):
        return url
    if m := re.match(
        r"^(?:https?://)?huggingface\.co/([^/]+)/([^/]+)(?:/([^/]+\.safetensors))?/?$",
        url,
    ):
        if len(m.groups()) not in [2, 3]:
            raise ValueError(
                "Invalid HuggingFace URL. Expected format: huggingface.co/<owner>/<model-name>[/<lora-weights-file.safetensors>]"
            )
        return url
    if m := re.match(r"^(?:https?://)?civitai\.com/models/(\d+)(?:/[^/?]+)?/?$", url):
        model_id = m.groups()[0]
        return make_civitai_download_url(model_id, civitai_api_token)
    if m := re.match(r"^((?:https?://)?civitai\.com/api/download/models/.*)$", url):
        return url
    if m := re.match(r"^(https?://.*\.safetensors)(?:\?|$)", url):
        return url  # might be signed, keep the whole url
    if m := re.match(r"^(https?://.*\.safetensors\?.*)$", url):
        return url  # URL with query parameters, keep the whole url
    if m := re.match(r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/?$", url):
        owner, model_name = m.groups()
        return make_replicate_model_download_url(owner, model_name)
    if m := re.match(
        r"^(?:https?://replicate.com/)?([^/]+)/([^/]+)/(?:versions/)?([^/]+)/?$", url
    ):
        owner, model_name, version_id = m.groups()
        return make_replicate_version_download_url(owner, model_name, version_id)
    if m := re.match(r"^(https?://replicate.delivery/.*\.tar)$", url):
        return m.groups()[0]

    if "huggingface.co" in url:
        raise ValueError(
            "Failed to parse HuggingFace URL. Expected huggingface.co/<owner>/<model-name>"
        )
    if "civitai.com" in url:
        raise ValueError(
            "Failed to parse CivitAI URL. Expected civitai.com/models/<id>[/<model-name>]"
        )
    raise ValueError(
        """Failed to parse URL. Expected either:
* Replicate model in the format <owner>/<username> or <owner>/<username>/<version>
* HuggingFace URL in the format huggingface.co/<owner>/<model-name>
* CivitAI URL in the format civitai.com/models/<id>[/<model-name>]
* Arbitrary .safetensors URLs from the Internet"""
    )


def make_replicate_model_download_url(owner: str, model_name: str) -> str:
    return f"https://replicate.com/{owner}/{model_name}/_weights"


def make_replicate_version_download_url(
    owner: str, model_name: str, version_id: str
) -> str:
    return f"https://replicate.com/{owner}/{model_name}/versions/{version_id}/_weights"


def make_civitai_download_url(
    model_id: str, civitai_api_token: str | None = None
) -> str:
    if civitai_api_token is None:
        return f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
    return f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor&token={civitai_api_token.get_secret_value()}"
