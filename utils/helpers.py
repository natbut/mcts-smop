
import numpy as np


def sample_from_range(min, max):
    if min == max:
        return min

    if isinstance(min, int):
        return np.random.randint(min, max)
    else:
        return np.random.uniform(min, max)
