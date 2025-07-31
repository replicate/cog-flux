import tempfile
import time
from pathlib import Path

import httpx

from helpers.utils.retry import retry_with_exponential_backoff

# Constants
MAX_RETRIES = 4
DOWNLOAD_RETRY_DELAY = 1
CHUNK_SIZE = 1024 * 1024  # 1MB chunks


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    output_format: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = DOWNLOAD_RETRY_DELAY,
    chunk_size: int = CHUNK_SIZE,
) -> Path:
    """
    Download a file from a URL with retry logic and validation.

    Args:
        client: httpx AsyncClient instance
        url: URL to download from
        output_format: File extension for the output file
        max_retries: Maximum number of download attempts
        retry_delay: Initial delay between retries (will be increased exponentially)
        chunk_size: Size of chunks to download in bytes

    Returns:
        Path to the downloaded file

    Raises:
        Exception: If download fails after maximum retries or if downloaded file is invalid
    """
    t = time.time()

    async def attempt_download() -> Path:
        total_size = 0
        with tempfile.NamedTemporaryFile(
            suffix=f".{output_format}", delete=False
        ) as tmp:
            file_path = Path(tmp.name)
            async with client.stream("GET", url, follow_redirects=True) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    print(
                        f"Download failed: status={response.status_code}, text={await response.aread()}\nException: {e}"
                    )
                    raise
                content_length = response.headers.get("content-length")
                expected_size = int(content_length) if content_length else None
                if expected_size:
                    print(f"Downloading {expected_size} bytes")

                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    tmp.write(chunk)
                    total_size += len(chunk)

            if expected_size and total_size != expected_size:
                message = f"Incomplete download: got {total_size} bytes, expected {expected_size} bytes"
                print(f"{message}")
                raise Exception(message)

            if total_size < 512:
                message = f"Downloaded file is too small: {total_size} bytes"
                print(f"{message}")
                raise Exception(message)

            print(
                f"Downloaded {total_size / 1024 / 1024:.2f}MB in {time.time() - t:.2f}sec"
            )
            return file_path

    retry_count = 0
    while retry_count < max_retries:
        try:
            return await attempt_download()
        except (httpx.HTTPError, Exception) as e:
            if isinstance(e, httpx.HTTPError):
                error_type = "HTTP Error"
            else:
                error_type = "Download Error"

            should_continue, retry_count = await retry_with_exponential_backoff(
                retry_count=retry_count,
                max_retries=max_retries,
                base_delay=retry_delay,
                error_type=error_type,
                error_message=f"Failed to download after {max_retries} attempts: {str(e)}",
            )
            if not should_continue:
                break

    raise Exception("Failed to download file after maximum retries")
