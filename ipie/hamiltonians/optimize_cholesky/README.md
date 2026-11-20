# optimize_cholesky

Cholesky gauge and partition optimization for linearized auxiliary-field QMC
(LAFQMC; Sorella, Phys. Rev. B 84, 241110(R), 2011).

LAFQMC writes the propagator as a positive combination of one-body
propagators, `G = Lambda - H = sum_i a_i U_i` with `a_i > 0`. This package
builds that decomposition from an exact Cholesky factorization of the
two-electron integrals. It chooses the free parameters so that the terms
commute as nearly as possible:

- an orthogonal **gauge** `O` that mixes the Cholesky vectors (the integrals
  are unchanged);
- one **partition** coordinate `t` per source, which splits a coupling
  between its two amplitudes;
- optionally, the spin-resolved one-body **shift** `(mu_up, mu_down)`.

The objective is the squared commutator loss between the terms, weighted by
the term weights by default. The result is saved as an HDF5 file that
`QCSOR.from_file` (in `ipie/hamiltonians/sor_base.py`) reads directly.

The full reference, with the mathematics, every option and the file format,
is [`docs/optimize_cholesky_documentation.pdf`](docs/optimize_cholesky_documentation.pdf).
Its ReportLab sources are in `docs/build/`; `python docs/build/build_doc.py out.pdf`
regenerates it. This README is the
in-repository guide: quick start, layout, how an evaluation runs, how to
extend and debug the solver, and how to test.

**At a glance**

- The directory has no `__init__.py`. Import every name from its submodule,
  for example
  `from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator`.
- The default objective is the weighted loss (`loss_kind="weighted"`)
  everywhere.
- All inner-solver settings travel in one `InnerOptions` object.
- `backend="pytorch"`, `solver="gdm"` and `GDMOptimizer` are reserved names.
  They raise `NotImplementedError` before doing any work; nothing silently
  falls back.

## Quick start

Inputs follow these conventions. They are checked, never repaired:
`prepare_problem` validates that every array is real and finite and that
the factors reproduce every ERI entry, and returns read-only float64 copies.

| Input | Shape | Convention |
| --- | --- | --- |
| `h1` | `(n, n)` | Bare one-body matrix; no normal-ordering correction is applied |
| `eri` | `(n, n, n, n)` | `eri[p, r, q, s] = (pr\|qs)` |
| `chol` | `(r, n, n)` | Real symmetric factors with `eri = einsum("gpr,gqs->prqs", chol, chol)` |

```python
import numpy as np
from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem
from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
from ipie.hamiltonians.optimize_cholesky.io import save_decomposition

problem = prepare_problem(h1, eri, chol, nelec=(2, 2))   # validates; picks the sector policy
engine = GaugeEvaluator(problem, tau1=0.03, tau2=0.04)   # loss_kind="weighted" by default

# Start from small random angles: the input factors (theta = 0) often have exact zero
# eigenvalues, where the balanced map has no derivative (GaugeRegularityError).
r = problem.data.nchol
theta0 = 0.1*np.random.default_rng(0).uniform(-1.0, 1.0, r*(r - 1)//2)

result = optimize_gauge(
    engine, theta0=theta0,
    options=LBFGSOptions(max_iterations=50, line_search="weak_wolfe", verbosity=1),
    inner_options=InnerOptions(optimize_shift=True),
)
print(result.status, result.initial_loss, result.loss, result.decomposition.shift)
save_decomposition(path, result, overwrite=True)
```

Then, for LAFQMC:

```python
from ipie.hamiltonians.sor_base import QCSOR

ham = QCSOR.from_file(path, nelec=(2, 2), thresh=None)   # thresh=None: no truncation (exact)
```

`QCSOR.optimize_from_cholesky(h1, eri, chol, nelec, tau1, tau2, path, ...)`
does both steps in one call. Its options are keyword-only, and an unknown
name raises `TypeError`. It starts from random angles of scale `theta`
(reproducible with `seed=`). It runs weak-Wolfe L-BFGS with the shift
relaxed and the `InnerOptions` defaults (1e-11 / 1e-9, 500 sweeps). The
outer defaults are 100 iterations, verbosity 2 and the outer tolerances
1e-8 / 1e-6. You can override:

- `maxiter`, `verbosity`, `gradient_atol`, `gradient_rtol` (outer loop);
- `maxsweep`, `inner_atol`, `inner_rtol`, `amplitude_limit`,
  `one_body_limit` (inner solve);
- whole `options=` / `inner_options=` objects.

The illustrative `tau1`, `tau2` are not universal. Choose them so the
amplitudes are feasible for your integrals. Two-body amplitudes need
`|d| < amplitude_bound <= 1`, and one-body amplitudes need `1 + d > 0`. A
smaller `tau2` makes more gauges feasible. Amplitudes are checked, never
clipped.

## Package layout

Each numerical module builds on the ones above it. `io.py` only reads
results, and `sor_base.py` (outside the package) only reads files.

| Module | Responsibility | Main entry points |
| --- | --- | --- |
| `models.py` | Records, options, exceptions, results; the array tables | `PreparedProblem`, `FixedDecomposition`, `SourceTable`, `ChannelTable`, `InnerOptions`, `LBFGSOptions`, `OuterOptimizationResult` |
| `problem.py` | Input validation, HDF5 loading, sector policy, gauges, factor eigensystems | `prepare_problem`, `load_problem`, `build_gauge_state`, `exponential_gauge`, `near_zero_eigenvalues` |
| `channels.py` | Sources, the branch table, the amplitude map, one- and two-body channel tables, rotation matrices | `enumerate_sources`, `build_fixed_decomposition`, `fixed_partition_amplitudes`, `with_channels`, `get_rotation_matrix` |
| `loss.py` | Mode-space commutator losses and their pullback | `objective_loss`, `raw_loss`, `commutator_loss`, `loss_pullback`, `ModeSpace` |
| `shift.py` | One-body shift bounds, exact quadratic profiles and updates | `with_shift`, `shift_coefficients`, `optimal_shift` |
| `partitions.py` | Partition intervals, source profiles, the inner solver | `optimize_partitions`, `partition_coefficients`, `with_partitions`, `partition_bounds` |
| `gradients.py` | Pullbacks, fixed / fixed-z / relaxed gauge gradients, the evaluator | `GaugeEvaluator`, `relaxed_gauge_gradient`, `check_gauge_regularity`, `check_gauge_direction` |
| `optimizers.py` | Exponential objective, solver-neutral parent, L-BFGS, GDM placeholder | `optimize_gauge`, `ExponentialGaugeObjective`, `LBFGSOptimizer` |
| `io.py` | The QCSOR-ready HDF5 file (format `lafqmc-optimized-decomposition`, version 2) | `save_decomposition` |

### Data model

A decomposition stores arrays, not per-channel objects:

- **`SourceTable`** holds the topology: factor, modes, kind and spins per
  source. It also holds the canonical channel layout: each source's channel
  range, and per channel its source, branch sign, spins and weight.
- **`ChannelTable`** and **`OneBodyTable`** hold the channels.
- **Record views.** `two_body.channels`, `one_body.channels`,
  `two_body.source_slices` and `PartitionOptimizationResult.coefficients`
  (a `PartitionProfiles`) behave as read-only sequences of the record types
  `Channel`, `OneBodyChannel`, `DensitySource` and `PartitionCoefficients`.
  The records are built on access and never on the optimization path.
- **Canonical and non-canonical.** The builders and the optimizer produce
  canonical decompositions: the table is the canonical layout of the sources
  at their amplitudes. `with_channels` accepts caller-supplied channel lists
  and returns a non-canonical decomposition. The loss functions, rotation
  matrices and Fock-space checks accept it; the partition solver, the
  gradients and `save_decomposition` reject it.

The weights `a_i` are positive and unnormalized. On the selected sector,
`(denom + mu_up N_up + mu_down N_down) I - H = sum_i a_i U_i`; with zero
shift this is `denom I - H`. The one-body and two-body lists share one
normalization, so never normalize them separately.

## How an evaluation runs

**Loss.** Every spin block of every channel is diagonal in one basis: a
rotated Cholesky factor or the `h` eigenbasis. The loss is evaluated in mode
space, over the `M = (r+1)n` modes:

- one dense overlap matrix `F` between modes of different bases (pairs
  within a basis commute);
- per-mode loads aggregated from the channel tables;
- for same-spin quartet legs, an off-diagonal term.

The cost does not depend on the number of channel pairs.

**Inner solve** (`optimize_partitions`, `GaugeEvaluator.relaxed`). At a
fixed gauge, given the loads of every other basis, the loss is affine in
one factor's loads. Every source of that factor therefore has the exact
profile `C + A exp(2t) + B exp(-2t)`, with minimizer `t* = log(B/A)/4`
clipped to its interval. A sweep updates the factors one at a time, which
gives the same iterates as a source-by-source Gauss-Seidel sweep. With
`optimize_shift=True` it then sets each `mu` to its exact minimizer, a
weighted mean of the orbital energies clipped to
`mu >= max(eps) - one_body_limit/tau1`.

The solve stops on the projected t-gradient (and, with the shift, on the
last mu change). A sweep limit is reported as `max_sweeps`, never as
convergence. No ERIs, eigendecompositions or rotations are rebuilt inside
the solve.

**Gauge gradient.** `GaugeEvaluator` offers three derivatives at a gauge
`O`. They differ in what is held fixed:

- `fixed(O, t=...)` holds `t` and the shift fixed;
- `fixed_z(O, z=...)` holds the normalized interval coordinates `z` in
  `[-1, 1]` fixed;
- `relaxed(O, options=...)` solves the inner problem and takes the envelope
  gradient, including the response of the gauge-dependent partition
  intervals.

The pullback chain is: loss to channel amplitudes, amplitudes to factor
eigenvalues and `t`, interval bounds, the eigendecomposition pullback for
all factors at once, and finally `O`. The inner sweeps are never
differentiated.

**Per trial gauge**, the evaluator:

- rotates and diagonalizes the factors once;
- runs `check_gauge_regularity` before any inner work, so near-degenerate
  factor eigenvalues, or near-zero ones for `balanced_log`, raise
  `GaugeRegularityError` straight away;
- builds the mode space once and shares it between the inner solve and the
  gradient.

The exponential coordinates form `expm` once per gauge. The public gradient
functions (`fixed_partition_gradient`, `normalized_partition_gradient`,
`relaxed_gauge_gradient`) also validate their input. They agree with the
evaluator, and with the frozen reference oracle, to the 1e-12 standard.

Inner and outer solutions are converged local stationary points on a
tracked branch, not certified global minima. Degeneracies raise
`GaugeRegularityError` instead of being regularized.

## Outer optimization

`ExponentialGaugeObjective` is the reduced objective
`F(theta) = L(expm(S(theta)) @ reference, t*(theta), mu*(theta))`. Here
`theta` packs the strictly upper triangle in the order
`(0,1), (0,2), ..., (1,2), ...`. Every trial gauge, including rejected
line-search trials, gets its own converged inner solve.

The objective caches one evaluation and keeps cumulative counters.
`inner_t0` is a frozen seed for every trial; with `None`, each trial starts
at its own interval midpoints. A rejected trial never seeds another.

`LBFGSOptimizer` is an explicit L-BFGS; no SciPy optimizer hides the
iterations. With no curvature history, or when the quasi-Newton direction
is not a finite descent direction, it takes a normalized steepest-descent
step. `max_step_norm` caps the first trial displacement. A curvature pair
is kept only when `s.y` is sufficiently positive.

Two line searches are available:

- `"armijo"` (default) backtracks until
  `F(theta + alpha p) <= F(theta) + armijo alpha g.p`.
- `"weak_wolfe"` is Lewis-Overton bracketing. It adds the one-sided
  curvature test `g(theta + alpha p).p >= wolfe_c2 g.p` and expands alpha up
  to `wolfe_max_step_norm`.

**Why weak Wolfe.** The reduced loss depends on `|ell_alpha ell_beta|`. It
therefore has a kink wherever an eigenvalue of a rotated factor crosses
zero, and minimizers often sit on kinks. There the directional derivative
jumps, so strong Wolfe has no solution. A point just beyond a kink satisfies
the weak test, gives `s.y > 0`, and lets the quasi-Newton model learn the
curvature across the kink. If no trial passes both tests, the lowest-loss
decreasing trial is accepted (`armijo_only`, or `expansion_capped`). Only
when no trial decreases the loss does the step return `line_search_failed`;
at a nonsmooth minimizer that is the expected exit. `near_zero_eigenvalues`
lists the active kinks.

An observation from before the steps-2 to 7 refactor (H2/STO-3G, six random
starts): weak Wolfe reached a lower loss than Armijo from five starts, used
1.4–3.9 times as many evaluations, and did not change basins. Use several
starting gauges when local minima matter.

**Stopping.** Both the theta-gradient infinity norm and the Frobenius norm
of the physical tangent gradient must fall below
`gradient_atol + gradient_rtol * (initial norm)`. With the defaults (1e-8,
1e-6) and typical initial gradients of 1e2 to 1e3, the rtol term dominates.
Pass `gradient_rtol=0` for a true absolute stop. The result always
describes the last accepted state.

| Status | Meaning |
| --- | --- |
| `converged` | Both gradient tolerances met |
| `coordinate_stalled` | The theta-gradient met its tolerance; the physical tangent gradient did not |
| `max_iterations`, `max_evaluations` | Budget exhausted |
| `line_search_failed` | No trial decreased the loss (expected at a nonsmooth minimizer) |
| `stalled` | No finite descent direction |
| `callback_stopped` | The callback raised `StopIteration` |

Inside a line search, `RECOVERABLE_TRIAL_ERRORS` reject the trial and
trigger backtracking: feasibility, regularity, inner convergence,
floating-point and linear-algebra errors. Any other exception propagates.
A failing initial evaluation raises.

### Extending the solver

`OuterOptimizer` knows neither a solver name nor a coordinate
representation. It owns budgets, callbacks, stopping and accepted-state
bookkeeping. A new solver supplies three hooks and its options record in
`models.py`, and registers in `optimize_gauge`:

| Hook | Must return |
| --- | --- |
| `_initialize(initial)` | `(initial_point, fresh_solver_state)`; initial failures propagate |
| `_step(point, run, iteration)` | `OptimizerStep(evaluation=accepted_point, ...)`, or a failure status with no evaluation |
| `_build_result(point, summary)` | The solver's result (a dataclass with a `validation_report` field) |

Points supply `loss`, `gradient_norm`, `tangent_gradient_norm` and
`gauge_gradient.inner_result`. A step that makes several trial evaluations
must respect `_budget_exhausted(run)`. `GDMOptimizer` marks where a
geometric solver would go.

### Debugging L-BFGS

| Location | What to inspect |
| --- | --- |
| `_lbfgs_direction` | Two-loop inverse-Hessian product and last-pair scaling |
| `LBFGSOptimizer._step` | Descent fallback, displacement cap, search dispatch, curvature acceptance |
| `LBFGSOptimizer._armijo_search` / `_weak_wolfe_search` | The two line searches |
| `ExponentialGaugeObjective.evaluate` | Gauge, inner convergence, coordinate pullback, cache, counters |
| `OuterOptimizer.optimize` | Accepted-state lifecycle, budgets, callbacks, stopping |

`result.history` (and the callback) receives the initial and every accepted
`OuterIteration`. `result.trials` records every attempted line-search point
with its reason. Look at `curvature_update`, `memory_size`,
`line_search_trials`, `inner_sweeps` and `inner_projected_gradient_norm`.
`verbosity=2` prints every trial, including the weak-Wolfe `slope_ratio`.

## Tests

`pytest.ini` selects only tests marked `unit`, `driver` or `mpi`. These
tests carry no markers, so clear the filter:

```sh
# everything (about 4.5 min, almost all of it the frozen oracle's pair-loop self-check)
python -m pytest ipie/hamiltonians/optimize_cholesky/tests -m ""
# everything except that self-check (about 15 s)
python -m pytest ipie/hamiltonians/optimize_cholesky/tests -m "" \
    --deselect ipie/hamiltonians/optimize_cholesky/tests/test_reference_golden.py::OracleGoldenTests
```

**Equivalence standard.** Comparisons between two code paths, or against
the frozen reference, use `tests/tolerance.py`:
`max|actual - expected| <= 1e-12 * max|expected|`. Gradient fields are
scaled by `max(max|field|, |loss|)`, because inner residuals are not a
meaningful scale. Exact comparisons are kept only where exactness is the
contract:

- integers, layouts and flags;
- structural zeros;
- copies and inputs that must not change;
- seeded reproducibility.

Patch-based tests assert behaviour contracts ("the inner loop never
re-diagonalizes", "a singular gauge fails before the inner solve"), not
numerical values.

**Frozen reference.** `tests/reference/` pins the behaviour of commit
`3680f29`:

- `reference_kernels.py`, a self-contained oracle;
- `golden_3680f29.npz`, golden numbers for 24 synthetic configurations and
  H2/6-31G, its triplet, H4/STO-3G, LiH/STO-3G and H2/cc-pVDZ;
- `benchmark.py`.

`test_reference_golden.py` replays every case through the public entry
points at 1e-12, including 5-iteration L-BFGS trajectories. Do not
regenerate the golden file; it is the yardstick.

**Modules** (224 tests):

- **Exactness and finite differences.** `test_channels` checks exact
  Fock-space reconstruction for every policy, map and gauge.
  `test_gradients` checks all three gauge gradients against finite
  differences, the moving bounds, the evaluator against the public functions
  and the oracle, and backends. `test_shift` covers shifted channels, Λ,
  profiles, relaxation order, bounds and relaxed gradients.
- **Loss, solver and objective.** `test_loss` compares the loss with the
  dense reference and `sor_base` diagnostics. `test_partitions` covers
  profiles, analytic updates, the inner solver and its errors.
  `test_weighted` covers the weighted objective, its profiles, inner solve
  and gradients.
- **Outer optimization.** `test_optimizers` covers the outer objective,
  L-BFGS on controlled and physical problems, solver classes and dispatch.
  `test_weak_wolfe` covers the weak-Wolfe search on smooth, kinked and
  physical objectives.
- **Inputs and data model.** `test_problem` covers input validation, HDF5
  loading, gauges and eigensystems. `test_tables` covers the array data
  model and its record views against the oracle.
- **Kernels.** `test_source_kernels` compares the batched per-source
  kernels with the step-3 loops (`tests/source_reference.py`) and covers the
  early regularity check and once-per-gauge work.
- **Files and wrapper.** `test_io_sor` covers the file format,
  `QCSOR.from_file`, truncation, walker energy and propagation.
  `test_io_arrays` checks that the save builds no record objects and that
  files at the amplitude limits load. `test_wrapper` covers the options of
  `QCSOR.optimize_from_cholesky`.

**Known failures.** Four walker subtests in `test_io_sor` fail on the
current `sor_base`/trial code, outside this package:

- `SingleDet.build` iterates `hamiltonian.h1` per spin, but `from_file`
  stores one `(n, n)` `h1`;
- the walker local energy now adds `ham.E0`, which `parse_decomposition`
  uses to carry the shift energy `mu.N` for Λ.

The walker tests skip when the LAFQMC walker modules are not importable,
and the HDF5 tests skip without h5py.

## Performance and memory

`tests/reference/benchmark.py --outer 20` times one relaxed evaluation and
20 L-BFGS iterations (2-core container):

| System | Relaxed evaluation | 20 L-BFGS iterations |
| --- | --- | --- |
| H2/cc-pVDZ (ab_only, n=10, r=54) | 0.050 s | 1.5 s |
| H4/6-31G (hybrid, n=8, r=32) | 0.224 s | 5.8 s |
| LiH/STO-3G (hybrid, n=6, r=21) | 0.044 s | 1.1 s |

The pair-loop implementation this package replaced was estimated at ~20 h
per relaxed evaluation for H2/cc-pVDZ. For hybrid systems the time is
dominated by `ModeSpace.orbital_K`. An optional patch from step 4
(`step4b_orbital_K.patch`) rewrites it as two GEMMs: H4 then takes 0.038 s
per evaluation.

**Memory.** The dense mode-space matrix `F` is `M x M` with `M = (r+1)n`.
That is about 2 MB for H2/cc-pVDZ and about 140 MB at n = 20 with a full
Cholesky, but about 1.6 GB at n = 30. Beyond n of about 20, `F` would have
to be applied blockwise, or fewer Cholesky vectors used.

## Changes from the previous README

- **Imports.** There are no package-level imports; import from the
  submodules. The old module names (`prepare`, `validation`, `objective`,
  `derivatives`, `outer`, `solvers.*`) are gone.
- **Default loss.** The default `loss_kind` is `"weighted"` everywhere (it
  was `"raw"`).
- **Loss API.** `OverlapEvaluator` and the `method=` argument of the loss
  functions were removed; losses are computed in mode space only. The pair
  kernels survive as test references (`tests/pair_reference.py`).
- **Decompositions.** They store array tables, and the record lists are
  read-only views. `dataclasses.replace(..., channels=...)` no longer works;
  use `channels.with_channels`. `PartitionOptimizationResult.coefficients`
  is a `PartitionProfiles`.
- **Files.** `save_decomposition` writes the QCSOR-ready HDF5 file directly
  from the tables, and `QCSOR.from_file` reads it.
- **Wrapper.** `QCSOR.optimize_from_cholesky` takes keyword-only options.
  Its inner defaults are now 1e-11 / 1e-9 with 500 sweeps (was 1e-6 / 1e-6,
  100).
