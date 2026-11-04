"""Input validation/preparation and configurable HDF5 loading for step 1."""

from numbers import Integral
from pathlib import Path

import numpy as np

from ipie.hamiltonians.optimize_cholesky.models import PreparedProblem, TensorData
from ipie.hamiltonians.optimize_cholesky.validation import _options, validate_inputs


def select_sector_policy(nelec, nbasis):
    """Return canonical particle counts and hybrid/ab_only/aa_only policy.

    Counts are optional in step 1; None selects the generic hybrid policy.
    A down-only sector is left generic rather than silently relabeling spins.
    """
    if nelec is None:
        return None, "hybrid"
    try:
        nelec = tuple(nelec)
    except TypeError as exc:
        raise ValueError("nelec must be (N_up,N_down)") from exc
    if len(nelec) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
        or not 0 <= v <= nbasis for v in nelec
    ):
        raise ValueError(f"nelec must contain two integers between 0 and {nbasis}")
    counts = tuple(int(v) for v in nelec)
    if counts[1] == 0:
        return counts, "aa_only"
    if counts == (1, 1):
        return counts, "ab_only"
    return counts, "hybrid"


def prepare_problem(h1, eri, chol, nelec=None, options=None):
    """Validate exact tensors and return owned, read-only float64 arrays.

    h1 is the original physical h, never h'. No normal-order correction is
    applied. No tensors are symmetrized or refactorized. No caller arrays
    are modified. eri uses the explicit axes (p,r,q,s), not (p,q,r,s).
    """
    options = _options(options)
    raw = TensorData(np.asarray(h1), np.asarray(eri), np.asarray(chol))
    report = validate_inputs(raw, options)
    counts, policy = select_sector_policy(nelec, raw.nbasis)
    arrays = [np.array(a, dtype=np.float64, order="C", copy=True)
              for a in (raw.h1, raw.eri, raw.chol)]
    for array in arrays:
        array.setflags(write=False)
    return PreparedProblem(TensorData(*arrays), report, counts, policy)


def _read_dataset(path, key):
    # Import lazily so array-only use does not require the optional I/O module.
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 loading requires h5py; install it with 'pip install h5py'") from exc
    path = Path(path)
    if not isinstance(key, str) or not key:
        raise ValueError("HDF5 dataset keys must be nonempty strings")
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Dataset {key!r} not found in {str(path)!r}")
        dataset = handle[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{key!r} in {str(path)!r} is not an HDF5 dataset")
        return dataset[()]


def load_problem(h1_path, eri_path=None, chol_path=None, *, h1_key="h1", eri_key="eri",
                 chol_key="chol", nelec=None, options=None):
    """Read one or three HDF5 files, close them, and call prepare_problem.

    If eri_path/chol_path are omitted, read those datasets from h1_path.
    Keys may include HDF5 groups, e.g. 'integrals/h1'. Full arrays are loaded
    into memory; this is not an out-of-core loader. No disk writes occur.
    """
    eri_path = h1_path if eri_path is None else eri_path
    chol_path = h1_path if chol_path is None else chol_path
    h1 = _read_dataset(h1_path, h1_key)
    eri = _read_dataset(eri_path, eri_key)
    chol = _read_dataset(chol_path, chol_key)
    return prepare_problem(h1, eri, chol, nelec=nelec, options=options)

