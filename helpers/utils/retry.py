import asyncio
import random
from enum import Enum, auto


class RetryType(Enum):
    EXPONENTIAL = auto()
    UNIFORM = auto()
    CAPPED_EXPONENTIAL = auto()


async def retry(
    retry_count: int,
    max_retries: int,
    base_delay: int,
    error_type: str = "Queued",
    error_message: str = "Queue is full. Please try again later.",
    retry_type: RetryType = RetryType.EXPONENTIAL,
    max_delay: int | None = None,
) -> tuple[bool, int]:
    if retry_count >= max_retries:
        raise Exception(error_message)

    if retry_type == RetryType.CAPPED_EXPONENTIAL and max_delay is None:
        raise ValueError("max_delay is required for CAPPED_EXPONENTIAL retry type")

    jitter = random.uniform(0.8, 1.2)

    if retry_type == RetryType.EXPONENTIAL:
        delay = base_delay * (2**retry_count) * jitter
    elif retry_type == RetryType.UNIFORM:
        delay = base_delay * jitter
    else:  # CAPPED_EXPONENTIAL
        delay = min(base_delay * (2**retry_count) * jitter, max_delay)

    print(f"{error_type}. Checking again in {delay:.1f} seconds...")
    await asyncio.sleep(delay)
    return True, retry_count + 1


async def retry_with_exponential_backoff(
    retry_count: int,
    max_retries: int,
    base_delay: int,
    error_type: str = "Queued",
    error_message: str = "Queue is full. Please try again later.",
) -> tuple[bool, int]:
    return await retry(
        retry_count=retry_count,
        max_retries=max_retries,
        base_delay=base_delay,
        error_type=error_type,
        error_message=error_message,
        retry_type=RetryType.EXPONENTIAL,
    )


async def retry_with_uniform_backoff(
    retry_count: int,
    max_retries: int,
    base_delay: int,
    error_type: str = "Queued",
    error_message: str = "Queue is full. Please try again later.",
) -> tuple[bool, int]:
    return await retry(
        retry_count=retry_count,
        max_retries=max_retries,
        base_delay=base_delay,
        error_type=error_type,
        error_message=error_message,
        retry_type=RetryType.UNIFORM,
    )


async def retry_with_capped_exponential_backoff(
    retry_count: int,
    max_retries: int,
    base_delay: int,
    max_delay: int,
    error_type: str = "Queued",
    error_message: str = "Queue is full. Please try again later.",
) -> tuple[bool, int]:
    return await retry(
        retry_count=retry_count,
        max_retries=max_retries,
        base_delay=base_delay,
        error_type=error_type,
        error_message=error_message,
        retry_type=RetryType.CAPPED_EXPONENTIAL,
        max_delay=max_delay,
    )
