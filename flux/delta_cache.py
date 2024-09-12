def add(x, y):
    if isinstance(x, tuple):
        return tuple([a + b for a, b in zip(x, y)])
    else:
        return x + y


def subtract(x, y):
    if isinstance(x, tuple):
        return tuple([a - b for a, b in zip(x, y)])
    else:
        return x - y


class DeltaCache:
    def __init__(self, start_step: int, end_step: int, start_block: int, end_block: int, cache_interval: int):
        self.start_step = start_step
        self.end_step = end_step
        self.start_block = start_block
        self.end_block = end_block
        self.cache_interval = cache_interval
        self.cache = {}
        self.last_update = {}

    def store(self, index, input, output, current_step):
        self.cache[index] = subtract(output, input)
        self.last_update[index] = current_step

    def retrieve(self, block_index, input, current_step):
        if (
            self.start_step <= current_step <= self.end_step
            and self.start_block <= block_index <= self.end_block
            and block_index in self.cache
            and block_index in self.last_update
            and current_step - self.last_update[block_index] < self.cache_interval
        ):
            return add(input, self.cache[block_index])
        return None

    def reset(self):
        self.cache = {}
        self.last_update = {}
