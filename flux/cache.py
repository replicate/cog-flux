from torch import Tensor
import torch 
import numpy as np

class TeaCache:
    def __init__(self):
        self.prev_step_input = None
        self.cached_residual = None
        self.accumulated_l1_delta = 0.0
        coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
        self.rescale_func = np.poly1d(coefficients)
    
    def should_use_cache(self, cur_step_input: Tensor, cache_threshold: float = 0.0):
        if self.prev_step_input is None:
            self.prev_step_input = cur_step_input
            return False
        
        l1_norm_delta = torch.sum((self.prev_step_input - cur_step_input).abs()) / torch.sum(self.prev_step_input.abs())
        self.prev_step_input = cur_step_input
        self.accumulated_l1_delta += self.rescale_func(l1_norm_delta)
        should_use_cache = l1_norm_delta < cache_threshold 
        if should_use_cache:
            self.accumulated_l1_delta = 0.0
        return should_use_cache

    def update_cache(self, new_residual: Tensor):
        self.cached_residual = new_residual