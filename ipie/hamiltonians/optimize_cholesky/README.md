# Cholesky optimization: step 1

This package implements input models, HDF5 loading, orthogonal Cholesky gauges,
factor diagonalization, and deterministic identity checks. It does not yet
generate density channels or optimize partitions/commutators, and it does not
modify `sor_base.py`.

Copy this directory into `ipie/hamiltonians/` (imports then start with
`ipie.hamiltonians.cholesky_optimization`), or put its parent on `PYTHONPATH`
and import `cholesky_optimization` directly. There are no imports from ipie.
Install NumPy and SciPy; also install h5py for HDF5 loading:

```sh
python -m pip install numpy scipy h5py
```

## Exact input conventions

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `h1` | `(n,n)` | Bare/original physical one-body matrix h, **not h'** |
| `eri` | `(n,n,n,n)` | `eri[p,r,q,s] = (pr|qs)` |
| `chol` | `(r,n,n)` | `chol[gamma,p,r]`, with `eri = einsum('gpr,gqs->prqs', chol, chol)` |

Require n>=1 and r>=1; zero-valued factors are allowed. Arrays must be real,
finite, and compatible in shape. h and every Cholesky
matrix must be symmetric within the explicit validation tolerance. Complex
dtypes are rejected even if their imaginary part is zero. Flattened/packed
or transposed storage formats are not guessed: reshape/convert them explicitly
before calling `prepare_problem` if needed. No tensor is symmetrized, fitted,
truncated, or given a normal-order correction here.

## Load HDF5 and evaluate a gauge

For one file containing datasets `h1`, `eri`, and `chol`:

```python
import numpy as np
from cholesky_optimization import (
    ValidationOptions, load_problem, exponential_gauge,
    givens_gauge, build_gauge_state,
)

options = ValidationOptions(atol=1e-12, rtol=1e-10, block_size=16)
problem = load_problem("integrals.h5", nelec=(1, 1), options=options)
initial = build_gauge_state(problem.data, options=options)  # O = identity

r = problem.data.nchol
theta = np.zeros(r * (r - 1) // 2)
if theta.size:
    theta[0] = 0.2
O = exponential_gauge(theta, r)
state = build_gauge_state(problem.data, O, options)

print(problem.policy)                 # ab_only for (1,1)
print(state.chol.shape)               # (r,n,n)
print(state.eigenvalues.shape)        # (r,n)
print(state.X.shape)                  # (r,n,n)
print(state.diagnostics.as_dict())    # absolute/relative errors and tolerances

# Alternative: a sequence of left-multiplied Givens rotations, angles in radians.
if r >= 2:
    O_givens = givens_gauge(r, [(0, 1, 0.2)])
    givens_state = build_gauge_state(problem.data, O_givens, options)
```

The files and dataset names are independently configurable:

```python
problem = load_problem(
    "h.h5", "eri.h5", "chol.h5",
    h1_key="integrals/h", eri_key="integrals/prqs", chol_key="factorization/L",
    nelec=(2, 0), options=options,
)
```

`eri_path` and `chol_path` default to `h1_path`. Keys can contain HDF5 group
paths. The loader opens files read-only and closes them before returning. It
loads full tensors into RAM; it is not an out-of-core loader. Preparation owns
read-only float64 copies, so peak memory includes the loaded tensors and their
prepared copies. In-memory data can instead be passed to
`prepare_problem(h1, eri, chol, nelec=..., options=...)`.

`nelec` is optional at this step. Without it the policy is `hybrid`; `(1,1)`
selects `ab_only`, and `N_down=0` selects `aa_only`. Counts are not guessed from
HDF5 datasets. Policy selection is recorded only: there are no generated
spin-dependent terms at step 1. The original h remains unchanged.

## Gauge and eigenvector conventions

- Exponential coordinates use `O = expm(S(theta)) @ reference`, with reference
  equal to identity by default. The strict upper triangle is packed in order
  `(0,1), (0,2), ..., (1,2), ...`; `S[p,q]=theta[k]`, `S[q,p]=-theta[k]`.
- `givens_gauge(r, [(p,q,angle), ...])` requires `p<q`. Its active block is
  `[[cos(angle), sin(angle)], [-sin(angle), cos(angle)]]`. Each rotation
  left-multiplies the current O in the supplied sequence; the last is leftmost.
- Mixing is `rotated[eta,p,r] = sum_gamma O[eta,gamma]*chol[gamma,p,r]`.
- A supplied orthogonal reference may have determinant -1. Neither validation
  nor rotation assumes determinant +1. The exponential map stays in the
  reference's determinant component.
- `state.X[gamma,p,alpha]` stores eigenvectors in columns. Eigenvalues are
  ascending. Reconstruction is
  `einsum('gpa,ga,gra->gpr', state.X, state.eigenvalues, state.X)`.
- All modes and factors, including zero or repeated ones, are retained.
  Eigenvector signs and bases within degenerate eigenspaces are not unique;
  tests compare reconstructed matrices, not arbitrary eigenvectors.

## Validation

`validate_inputs` and `validate_gauge_state` return a `ValidationReport` on
success and raise `ValidationError`, a `ValueError` subclass, on numerical
failure. The exception's `report` contains numerical metrics when available.
Malformed arrays fail before these matrix metrics can be constructed.

Acceptance for a numerical identity is
`max_abs_error <= atol + rtol * reference_max_abs`. Each factor gets its own
scale for symmetry and eigenpair reconstruction; a large factor cannot hide
an asymmetric small factor. Relative Frobenius errors are also recorded.
For zero reference norm the relative diagnostic is 0 for exact agreement,
otherwise `None`; absolute error determines acceptance.

The checks include:

1. Original h/factor symmetry and exact recovery of supplied ERIs.
2. Orthogonality of O, including reflections.
3. Agreement with the specified rotation, retaining (r,n,n).
4. `X.T @ X = I` and eigenpair reconstruction for every factor.
5. Rotated-factor recovery of supplied ERIs and direct ERI gauge invariance.
6. Invariance of `sum_gamma L_gamma @ L_gamma` (matrix products).

Every ERI element is checked. Using `C = chol.reshape(r,n*n)`, each comparison
evaluates row blocks of `C.T @ C`, avoiding an additional full n^4 reconstruction.
`block_size` controls the number of pair rows per block. This still requires
full ERI input storage and O(r*n^4) arithmetic per full comparison. Full checks
are appropriate at initialization and for reference/final verification, not
inside every future optimizer loss evaluation. Low-level `rotate_cholesky`
and `diagonalize_factors` do not reconstruct ERIs.

Defaults assume float64-quality exact data. If tensors were stored in float32,
explicitly choose tolerances matching that precision; converting to float64
does not recover lost digits. Never loosen tolerances to mask an index-order
or physical-convention error.

## Channel contract for the next step

`Channel(bix, spins, a_i, modes, d)` has exactly the requested fields:

```python
from cholesky_optimization import Channel
channel = Channel(
    bix=0, spins=(0, 1), a_i=5.0, modes=(2, 1), d=(0.1, -0.2)
)
```

`bix` indexes the **final** Cholesky eigenbasis, 0=up and 1=down; `d` contains
actual branch amplitudes, and `a_i` is the positive **unnormalized** coefficient
in Lambda*I-H. All four spin orientations are representable. A future adapter
can canonicalize `(1,0)` to `(0,1)` by swapping both legs, since opposite spins
commute. Mode and basis upper bounds, source identities, paired-domain
constraints, and branch-sharing checks belong to channel construction in
the next step. No channel is created by the loader or gauge builder now.

The `GaugeState` object is the completed output of step 1. It holds owned
read-only arrays and diagnostics, with no optimizer backend or open HDF5
handle. A final optimized/channel result and its persistence will be added
later; the present state does not imply optimization has occurred.

## Run tests

From the directory containing `cholesky_optimization/`:

```sh
python -m unittest discover -s cholesky_optimization/tests -v
```

The tests use independent dense contractions and temporary HDF5 files,
including custom dataset paths and separate input files.
When h5py is missing, the three actual HDF5 tests are explicitly skipped;
they execute normally after installing h5py. The array-only functions do not
import h5py. The present implementation is a NumPy/SciPy reference; a later
automatic-differentiation implementation must preserve these conventions
without passing differentiable tensors through NumPy conversions.

