from __future__ import division, print_function
from time import time

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


def mrmr(df, y, select=100, bins=10):
    df = np.array(df)
    y_bins = len(np.unique(y))

    mrmr = np.zeros((df.shape[1], 2))
    selected = []
    best = (-1, -np.inf)
    for j in xrange(df.shape[1]):
        mrmr[j, 0] = mi(df[:, j], y, bins=[bins, y_bins])
        if mrmr[j, 0] > best[1]:
            best = (j, mrmr[j, 0])
    selected.append(best[0])
    scores = [best[1]]
    while len(selected) < select:
        best = (-1, -np.inf)
        for j in xrange(df.shape[1]):
            if j in selected:
                continue
            mrmr[j, 1] += mi(df[:, j], df[:, selected[-1]], bins=bins)
            score = mrmr[j, 0] - mrmr[j, 1] / len(selected)
            if score > best[1]:
                best = (j, score)
        selected.append(best[0])
        scores.append(best[1])
    return selected


def mrmr_pool(df, y, select=100, pool_size=100, bins=10, verbose=False):
    df = np.array(df)
    y_bins = len(np.unique(y))

    select = min(select, df.shape[1])

    pool_increase = pool_size
    pool_size = min(df.shape[1], pool_size)

    if verbose:
        max_len = len(str(df.shape[1]))
        print('{:<{}} {:<{}} {}'.format('Feat', max_len, 'Pool', max_len, 'Time'))
        t0 = time()

    # Compute relevance
    rel_red = sorted(((mi(df[:, i], y, bins=[bins, y_bins]), 0, i) for i in xrange(df.shape[1])),
                     reverse=True)

    # Take bound
    bound = rel_red[pool_size][0]

    # Select first feature
    selected = [rel_red[0][2]]
    scores = [rel_red[0][0]]
    pool_sizes = [pool_size]

    if verbose:
        print('{:<{}d} {:<{}d} {:.0f}'.format(selected[-1], max_len,
                                              pool_size, max_len,
                                              time() - t0))

    rel_red.pop(0)
    pool_size -= 1

    while len(selected) < select:
        best = (-1, -np.inf)
        for i, (rel, red, f) in enumerate(rel_red):
            # update redundance
            red += mi(df[:, f], df[:, selected[-1]], bins=bins)
            rel_red[i] = (rel, red, f)
            score = rel - red / len(selected)
            if score > best[1]:
                best = (i, score)
        while best[1] < bound and len(selected) + pool_size < df.shape[1]:
            # increase pool
            start = pool_size
            pool_size = min(pool_size + pool_increase, df.shape[1] - len(selected))
            for i in xrange(start, pool_size):
                rel, red, f = rel_red[i]
                for j in selected[1:]:
                    red += mi(df[:, f], df[:, j], bins=bins)
                score = rel - red / len(selected)
                if score > best[1]:
                    best = (i, score)
            if pool_size >= len(rel_red):
                bound = rel_red[pool_size][0]
            else:
                bound = -np.inf
        # select best
        selected.append(rel_red[best[0]][2])
        scores.append(best[1])
        pool_sizes.append(pool_size)
        rel_red.pop(best[0])
        pool_size -= 1

        if verbose:
            print('{:<{}d} {:<{}d} {:.0f}'.format(selected[-1], max_len,
                                                  pool_size, max_len,
                                                  time() - t0))

    return selected
