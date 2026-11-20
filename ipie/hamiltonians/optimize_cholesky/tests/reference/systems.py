"""Inputs shared by the golden recorder, the golden tests and the benchmark.

synthetic(n, r, seed) is a frozen copy of tests/test_channels.example_tensors
at 3680f29, so the golden data never depends on a test module that may change.
molecule(name) reads systems.npz (written by make_systems.py; no PySCF needed).
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS_PATH = os.path.join(HERE, "systems.npz")
GOLDEN_PATH = os.path.join(HERE, "golden_3680f29.npz")


def synthetic(n=3, r=2, seed=23):
    """(h1, eri, chol) with eri[p,r,q,s] = sum_g chol[g,p,r] chol[g,q,s]."""
    rng = np.random.default_rng(seed)
    chol = []
    for g in range(r):
        X, _ = np.linalg.qr(rng.normal(size=(n, n)))
        ell = np.linspace(-0.45, 0.65, n) + 0.07*g
        chol.append((X*ell) @ X.T)
    chol = np.asarray(chol)
    eri = np.einsum("gpr,gqs->prqs", chol, chol)
    h = rng.normal(size=(n, n))*0.2
    h = (h+h.T)/2
    return h, eri, chol


_SYSTEMS = None


def molecule(name):
    """dict(h1, eri, chol, nelec, ecore, ehf, description) for one stored molecule."""
    global _SYSTEMS
    if _SYSTEMS is None:
        with np.load(SYSTEMS_PATH) as data:
            _SYSTEMS = {key: data[key] for key in data.files}
    prefix = f"{name}/"
    out = {key[len(prefix):]: value for key, value in _SYSTEMS.items() if key.startswith(prefix)}
    if not out:
        raise KeyError(f"No system {name!r} in {SYSTEMS_PATH}")
    out["nelec"] = tuple(int(v) for v in out["nelec"])
    out["description"] = str(out["description"])
    return out


def molecule_names():
    molecule("h2_631g")
    return sorted({key.split("/")[0] for key in _SYSTEMS})


def theta(nchol, scale, seed):
    """Deterministic exponential-gauge parameters, uniform in [-scale, scale]."""
    return scale*np.random.default_rng(seed).uniform(-1.0, 1.0, nchol*(nchol-1)//2)
