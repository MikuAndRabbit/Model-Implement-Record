import math
from typing import Dict, Optional
import numpy as np
from loguru import logger


def deprecated_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs = 0,
                     start_warmup_value = 0, warmup_steps = -1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    logger.info("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cosine_scheduler(base_value, final_value, max_length: int, warmup_length = 0, start_warmup_value = 0):
    warmup_schedule = np.array([])
    if warmup_length != 0:
        logger.info("Set warmup steps = %d" % warmup_length)
    if warmup_length > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_length)

    iters = np.arange(max_length - warmup_length)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == max_length
    return schedule


class EMA():
    def __init__(self, mu, shadow: Optional[Dict] = None):
        self.mu = mu
        self.shadow = {} if shadow is None else shadow

    def register(self, name, val):
        self.shadow[name] = val.clone()
        
    def get(self, name):
        return self.shadow[name]
    
    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
