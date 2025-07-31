import os


def seed_or_random_seed(seed: int | None) -> int:
    # Max seed is 2147483647
    if seed is None or seed <= 0:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

    print(f"Using seed: {seed}")
    return seed
