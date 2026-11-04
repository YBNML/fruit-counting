"""Seeded noise for the prediction path: one generator per integer key (`noise_rng`),
pass bias and depth noise. Every draw is made whatever the factor values, so one key
gives the same draws for every factor value.
"""

from __future__ import annotations

import numpy as np

from counting.data.formats.orchard_synth import PassKey

STEP_DET, STEP_POSE, STEP_BIAS, STEP_DEPTH = 1, 2, 3, 4


def noise_rng(*key: int) -> np.random.Generator:
    """A generator seeded by the integer `key`, one generator per distinct key."""
    if any(k < 0 for k in key):
        raise ValueError(f"noise key {list(key)} has a negative part; seeds and pitches must be >= 0")
    return np.random.default_rng([int(k) for k in key])


def pass_key_code(key: PassKey) -> tuple[int, int, int]:
    """`PassKey` as the integer parts used in a noise key."""
    return (round(key.pitch_deg * 1000), 0 if key.side == "A" else 1, key.aisle)


def pass_bias(rng: np.random.Generator, b: float) -> np.ndarray:
    """A length-`b` vector in a random direction, shape (3,)."""
    v = rng.standard_normal(3)
    return b * v / np.linalg.norm(v)


def perturb_depth(z: np.ndarray, rng: np.random.Generator, rel_sigma: float, dropout: float) -> np.ndarray:
    """`z` scaled by relative Gaussian noise, with a `dropout` fraction set to NaN. Does not mutate `z`."""
    z = np.asarray(z, dtype=np.float64)
    g = rng.standard_normal(z.shape[0])
    u = rng.random(z.shape[0])
    out = z * (1 + rel_sigma * g)
    out[u < dropout] = np.nan
    return out
