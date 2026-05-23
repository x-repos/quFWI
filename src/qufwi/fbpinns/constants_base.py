"""
Defines a generic base class which is inherited by the Constants class

This module is used by constants.py
"""

import os
import pickle

from qufwi.fbpinns.util import io


class ConstantsBase:

    # note can set members freely, below only for index assignment
    def __getitem__(self, key):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        return getattr(self, key)
    def __setitem__(self, key, item):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        setattr(self, key, item)

    def __str__(self):
        import numpy as np
        s = repr(self) + '\n'
        for k in vars(self):
            v = self[k]
            if k.endswith("_init_kwargs") and isinstance(v, dict):
                summary = {dk: f"array(shape={dv.shape})" if isinstance(dv, np.ndarray) else dv for dk, dv in v.items()}
                s += f"{k}: {summary}\n"
            else:
                s += f"{k}: {v}\n"
        return s

    # below methods assume self.run exist

    # calculated variables
    results_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "rasht")

    @property
    def summary_out_dir(self):
        return os.path.join(self.results_base, "summaries", self.run) + "/"
    @property
    def model_out_dir(self):
        return os.path.join(self.results_base, "models", self.run) + "/"

    def get_outdirs(self):
        io.get_dir(self.summary_out_dir)
        io.clear_dir(self.summary_out_dir)
        io.get_dir(self.model_out_dir)
        io.clear_dir(self.model_out_dir)

    def save_constants_file(self):
        "Save a constants to file in self.summary_out_dir"
        # Note: pickling only saves functions/ classes / modules by name reference so
        # the unpickling environment needs access to the source code
        # https://docs.python.org/3.7/library/pickle.html#what-can-be-pickled-and-unpickled
        with open(self.summary_out_dir + f"constants_{self.run}.txt", 'w') as f:
            for k in vars(self): f.write(f"{k}: {self[k]}\n")
        with open(self.summary_out_dir + f"constants_{self.run}.pickle", 'wb') as f:
            pickle.dump(vars(self), f)

    @property
    def constants_file(self):
        return self.summary_out_dir + f"constants_{self.run}.pickle"


def print_c_dicts(c_dicts):
    "Pretty print a list of c_dicts"

    # get full list of keys
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)

    for k in keys:
        print(f"{k}: ",end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print(f"{item}",end="")
            else: print(f"{item} | ",end="")
        print("")