from typing import Optional
import numpy as np
from brian2 import Quantity, meter


def generate_circle_locations(n, radius, rng):
    if isinstance(radius, Quantity):
        radius /= meter
    loc = np.zeros((n, 2))
    mask = np.ones(n, bool)
    while mask.sum() > 0:
        loc[mask] = rng.uniform(-radius, radius, (mask.sum(), 2))
        mask[mask] = np.sum(loc[mask]**2, axis=1) > radius**2
    return loc.T*meter


def get_distance(xPre, yPre, xPost, yPost):
    return np.sqrt((xPre - xPost)**2 + (yPre - yPost)**2)


def get_boxcar_connections(xPre, yPre, xPost, yPost, radius,
                           rng : Optional[np.random.Generator] = None,
                           probability=1,
                           outdegree=None):
    nPre, nPost = len(xPre), len(xPost)
    assert len(yPre) == nPre and len(yPost) == nPost
    iPre = np.repeat(np.arange(nPre), nPost)
    iPost = np.tile(np.arange(nPost), nPre)
    dist = get_distance(xPre[iPre], yPre[iPre], xPost[iPost], yPost[iPost])
    mask = (dist < radius) & (dist > 0)
    mask_2D = mask.reshape(nPre, nPost)  # a writable view
    assert probability >= 1 or outdegree is None, 'Specify either probability or outdegree, not both'
    if probability < 1:
        assert rng is not None, 'Provide an RNG for stochastic connectivity.'
        probe = rng.uniform(size=mask.shape)
        mask &= probe < probability
    elif outdegree is not None:
        assert rng is not None, 'Provide an RNG for stochastic connectivity.'
        for pre in range(nPre):
            candidates = np.flatnonzero(mask_2D[pre])
            if candidates.size > outdegree:
                nontargets = rng.choice(candidates, candidates.size - outdegree, replace=False, shuffle=False)
                mask_2D[pre, nontargets] = 0
    return iPre[mask], iPost[mask], dist[mask]