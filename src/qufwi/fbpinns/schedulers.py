"""
Defines active schedulers

Active schedulers are iterables which allow us to define which FBPINN subdomains
are active/fixed at each training step.

Each scheduler must inherit from the ActiveScheduler base class.
Each scheduler must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import numpy as np


class ActiveScheduler:
    """Base scheduler class to be inherited by different schedulers"""

    def __init__(self, all_params, n_steps):
        self.n_steps = n_steps
        self.m = all_params["static"]["decomposition"]["m"]
        self.xd = all_params["static"]["decomposition"]["xd"]

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        """
        Returns None if active array not to be changed, otherwise active array.
        active is an array of length m, where each value corresponds
        to the state of each model (i.e. subdomain), which can be one of:

        0 = inactive (but still trained if it overlaps with active models)
        1 = active
        2 = fixed
        """

        raise NotImplementedError


class AllActiveSchedulerND(ActiveScheduler):
    "All models are active and training all of the time"

    def __iter__(self):
        for i in range(self.n_steps):
            if i == 0:
                yield np.ones(self.m, dtype=int)
            else:
                yield None
