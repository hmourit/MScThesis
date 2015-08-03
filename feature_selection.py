from __future__ import division, print_function

import numpy as np
from itertools import product


def mi(x, y, bins=10):
    """Computes Mutual Information between two vectors."""
    pxy = np.histogram2d(x, y, bins=bins)[0]
    pxy = pxy / pxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for x, y in product(xrange(len(px)), xrange(len(py))):
        if pxy[x, y] != 0 and px[x] != 0 and py[y] != 0:
            mi += pxy[x, y] * np.log(pxy[x, y] / (px[x] * py[y]))
    return mi
