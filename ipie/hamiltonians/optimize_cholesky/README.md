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
Its ReportLab source is `docs/build_documentation.py`:
`python docs/build_documentation.py [out.pdf]` regenerates it (reportlab
needed; the DejaVu fonts are used when found). This README is the
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

**Energies after loading.** On the `(N_up, N_down)` sector,
`sum_i a_i U_i = Lambda I - H` with
`Lambda = ham.denom + mu.N + ham.cholesky_energy`, where `mu = ham.mu` is the
one-body shift and `cholesky_energy` is `E_c` (0 without a Cholesky shift).
`lambda_electronic` in the file is the untruncated Λ. `ham.E0` is the
constant added to a local energy measured with `ham.h1`. The energy
integrals (`ham.h1`, `v0` and the factor integrals `W = ell ell^T`) are always
the physical, unshifted ones, even when the channels carry a Cholesky
shift, so `from_file` leaves `E0 = 0`. `ham.h1` has shape `(2, n, n)`, one
copy per spin, as `decompose_h1` produces.

**Cholesky shifts in the file** (format version 3):
- the channels are stored shifted, with per-spin one-body bases (the
  eigenbases of `h_s`);
- the file adds `cholesky_shift`, `cholesky_energy` and the shifted channel
  eigenvalues `chol_lam`, which the truncation rule compares;
- `ham.cholesky_shift` and `ham.cholesky_energy` are set on load;
- a shifted file is exact only on the sector it was built for, so `from_file`
  refuses any other `nelec`.

Version 2 files still load. `make_shift` is a separate preprocessing route
and is unchanged.

`QCSOR.optimize_from_cholesky(h1, chol, nelec, tau1, tau2, path, ...)`
does both steps in one call. It builds `eri = einsum("gpr,gqs->prqs", chol, chol)`
itself, so the decomposition is exact for the Hamiltonian that `chol`
represents. Its options are keyword-only, and an unknown name raises
`TypeError`. It starts from random angles of scale `theta`
(reproducible with `seed=`). It runs weak-Wolfe L-BFGS with the shift
relaxed and the `InnerOptions` defaults (1e-11 / 1e-9, 500 sweeps). The
outer defaults are 100 iterations, verbosity 2 and the outer tolerances
1e-8 / 1e-6. By default it also optimizes a tied Cholesky shift from the
median start (`optimize_cholesky_shift="tied"`), so the file is exact only on
the `nelec` sector. Pass `optimize_cholesky_shift=False` for the earlier
behaviour, with no shift; that is also the setting to use without `nelec`.
You can override:

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
| `optimizers.py` | Exponential objective (gauge angles and, optionally, the Cholesky shift), solver-neutral parent, L-BFGS, GDM placeholder | `optimize_gauge`, `ExponentialGaugeObjective`, `LBFGSOptimizer`, `median_cholesky_shift` |
| `io.py` | The QCSOR-ready HDF5 file (format `lafqmc-optimized-decomposition`, version 3; version 2 still loads) | `save_decomposition` |

`tests/` holds the test modules, the shared tolerance and the frozen
reference; `docs/` holds the PDF and its source.

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

### Cholesky shifts

`build_fixed_decomposition(..., cholesky_shift=mu)` and
`GaugeEvaluator(..., cholesky_shift=mu)` shift the rotated factors,
`L'_{g,s} = L_g - mu[g,s] I`. `mu` is `(r,)` for a tied shift (both spins;
up only for aa_only) or `(r, 2)` per spin, and needs particle numbers. With
`m_g = sum_s N_s mu[g,s]`:

- the channels use `lambda[s,g,a] = ell[g,a] - mu[g,s]`; eigenvectors, and
  so the factor-factor overlaps, do not change;
- the one-body channels diagonalize `h_s = h + sum_g (m_g - mu[g,s]) L_g`
  (a tied shift: `h + (N-1) sum_g mu_g L_g`);
- `cholesky_energy` is `E_c = 1/2 sum_g (sum_s N_s mu[g,s]^2 - m_g^2)`, and
  `lambda_electronic = denom + mu.N + E_c`. The weights, and so `denom`, do
  not change. With a shift every policy is exact only on the sector.

Zero and tied shifts keep today's layouts and code path (`ell` replaced by
`lambda`). A split shift (`mu_up != mu_down` somewhere) stores per-spin
amplitudes `two_body.amplitudes` of shape `(sources, 2 legs, 2 spins)`, a
per-spin one-body eigensystem `one_body.X` `(2, n, n)`, and a per-spin h
strip in `ModeSpace`. A quartet keeps one `t`: its legs scale as
`lambda_{a,s} e^t` and `lambda_{b,s'} e^-t`. With `balanced_log` each
quartet leg is balanced by the RMS of its two spins' `lambda`, which equals
`|lambda|` when the spins agree.

The inner solver, the losses, the profiles and the gauge gradients accept
shifted decompositions, and `optimize_gauge` can optimize the shift together
with the gauge ("Optimizing the Cholesky shift" below). `save_decomposition`
stores it (format version 3) and `QCSOR.from_file` loads it; see "Energies
after loading". `QCSOR.optimize_from_cholesky` takes
`optimize_cholesky_shift="tied"` (the default) `| "split" | False` and
`cholesky_shift=None | "median" | "zero" | array`. With optimization, that
array is the start (`None`: the median); with `False` it is the fixed
shift (`None`: no shift).

**Gradients with a shift.** `h_s` depends on the gauge through
`L_g(O) = sum_h O[g,h] L0_h`, so `gradient_O` gains the term
`sum_s (m_g - mu[g,s]) <dL/dh_s, L0_h>`, where `dL/dh_s` is the eigh pullback
of `h_s` (eigenvalue adjoint `-tau1 dL/dd` per one-body channel, eigenvector
adjoint from the loss pullback). Every gradient also reports
`cholesky_gradient`, the derivative with respect to the shift at the same
fixed `t`, fixed `z` or envelope:

    dL/dmu[g,s] = -sum_a dL/dlambda[s,g,a] + N_s sum_s' P[g,s'] - P[g,s],
    P[g,s] = <dL/dh_s, L_g(O)>.

- `dL/dlambda` includes the moving interval ends. It is per spin: quartet
  legs are differentiated through their RMS leg scale.
- A relaxed one-body shift held on its lower bound `max(eps_s) - rho_1/tau1`
  moves with `h_s`; that term goes to `bound_correction_O`.
- The Cholesky shift is an outer variable: the inner solve keeps it fixed,
  so its relaxed gradient is an ordinary partial derivative at the envelope.

`cholesky_gradient=None` (the default, on `GaugeEvaluator` and on the three
public gradient functions) reports it in the shift's own form: `(r,)` for a
tied shift (column 0 for aa_only, whose `mu_down` is 0), `(r, 2)` for a split
one, nothing without a shift. `"tied"` or `"split"` choose the form, also at
zero shift. `"split"` keeps per-spin one-body bases (equal ones at a tied
shift). The tied gradient is the sum of the split columns.

Two regularity notes:

- **Degenerate h.** A gradient through `h_s` needs simple eigenvalues of
  `h_s`. A repeated one raises `GaugeRegularityError` ("One-body matrix
  h_sigma ...") before any inner work. Bare `h` is degenerate for
  H2/cc-pVDZ and LiH, so a shift gradient at zero shift is undefined there;
  a nonzero shift usually lifts the degeneracy.
- **Kink at equal spins.** A quartet leg's interval end takes the larger of
  its two spins' amplitudes. At a tied shift the two are equal, so the
  fixed-z loss has a kink in each spin's direction (not in the tied one), and
  so does the relaxed loss when a quartet source sits on an amplitude wall.
  The split gradient there is the mean of the two one-sided derivatives.

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
  gradient;
- with a Cholesky shift (or a shift gradient), diagonalizes `h_s` and checks
  its eigengaps in the same early step.

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

### Optimizing the Cholesky shift

`optimize_gauge(..., optimize_cholesky_shift="tied" | "split")` optimizes
the gauge and the Cholesky shift together:

```python
result = optimize_gauge(evaluator, theta0=theta0,
                        optimize_cholesky_shift="tied",    # or "split"
                        cholesky_shift0="median",          # default; "zero" or an array
                        options=LBFGSOptions(line_search="weak_wolfe"),
                        inner_options=InnerOptions(optimize_shift=True))
result.cholesky_shift          # final (r, 2) shift
result.initial_cholesky_shift  # the start
result.cholesky_scale          # s, see below
```

**Coordinates.** The solver works on `x = (theta, mu/s)`. A tied shift has
`r` entries; a split one has `2r`, all spin-up entries first. aa_only always
has `r`, because its `mu_down` is 0. The scale `s_g` is the eigenvalue
spread of rotated factor `g` at the starting gauge (`cholesky_shift_scale`),
fixed for the run, so that one unit of `x` moves `mu_g` across its factor's
spectrum. `ExponentialGaugeObjective(optimize_cholesky_shift=...)` exposes
`pack` and `unpack` for building `x` and reading it back. Its cache is keyed
on `x`.

**Start.** `"median"` is each rotated factor's spectral median at
`O(theta0)` (`median_cholesky_shift`), for both spins. For an even number
of orbitals this is `np.median`, as in `sor_base.compute_cholesky_shift`.
For an odd number it is the midpoint of the wider gap next to the middle
eigenvalue, because `np.median` would put `lambda = 0` exactly on that
mode.

**Modes.**

| Call | Optimizes |
| --- | --- |
| `optimize_cholesky_shift=False` (default) | `theta`, with the evaluator's own shift |
| `optimize_cholesky_shift=False, cholesky_shift0=...` | `theta`, at that fixed shift |
| `optimize_cholesky_shift="tied"` or `"split"` | `theta` and `mu` |
| ... plus `optimize_theta=False` | `mu` only, with the gauge held at `theta0` |

**Stopping.** `gradient_norm` is the infinity norm over all of `x`.
`tangent_gradient_norm` combines the physical tangent norm and the 2-norm of
the scaled shift gradient (the shift gradient alone when the gauge is
frozen).

**Tied or split?** Prefer `"tied"`. When `N_up = N_down` the loss is
unchanged by swapping the spins together with `mu_up <-> mu_down`, so a tied
point is stationary in every split direction and a gradient method started
there never splits. On H2 (ab_only), H4 and LiH, at N_up = N_down and at
N_up != N_down, a split continuation at a fixed gauge ended within 0.012% of
a tied one, while costing 9% to 160% more per evaluation.
`tests/reference/spin_split_benchmark.py` repeats that measurement.

**Kinks in mu.** The relaxed loss has a kink wherever `mu[g,s]` meets an
eigenvalue of factor `g` (`lambda = 0`), and its minima tend to sit there.
`near_zero_eigenvalues` reports the shifted eigenvalues as
`(factor, mode, lambda, spin)`. `balanced_log` cannot be differentiated at
`lambda = 0`: such trials are rejected and the line search backtracks.
`anchored_log` can. Away from the amplitude walls both maps give the same
relaxed loss, because they differ only by a per-source offset in `t`.

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
# everything (about 8 min, almost 7 of them the frozen oracle's pair-loop self-check)
python -m pytest ipie/hamiltonians/optimize_cholesky/tests -m ""
# everything except that self-check (about 1.5 min)
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

**Modules** (281 tests):

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
  `QCSOR.optimize_from_cholesky` and that it builds `eri` from `chol`.
  `test_make_shift` checks `QCSOR.make_shift` in Fock space: the bare form
  it optimizes `mu` for, and the `h'` form and `E0` it returns.
- **Cholesky shifts.** `test_cholesky_shift` checks the shifted forward model:
  Fock-space exactness for every policy and map with tied and split shifts,
  zero shifts against today's decomposition, a tied shift against shifted
  inputs, the per-spin path against the shared one at equal spins, the
  mode-space loss against the dense loss, exact profiles, intervals, the
  inner solve, validation and the step-12 saving guard.
  `test_cholesky_gradients` checks the gradients with a shift: finite
  differences in `O` and `mu` for all three modes, every policy, both maps,
  tied and split; a one-body shift on its wall and sources on amplitude
  walls; tied against split; zero shift; the kink at equal spins; public
  functions against the evaluator; regularity and validation.
  `test_io_cholesky_shift` covers shifted files: attributes and physical
  integrals, the QCSOR against a per-channel reference, the sector
  Hamiltonian and Λ, the sector check, truncation by each channel's shifted
  magnitude, the walker local energy and same-seed propagation, version 2
  files, the reader's checks, and the wrapper's shift options end to end.
  `test_cholesky_shift_optimization` covers the outer optimization over `(theta, mu/s)`:
  coordinate layouts, the median start and the scale, finite differences in
  `x`, the cache and stopping norms, every `optimize_gauge` mode (theta-only
  runs unchanged), exactness of the result, and `near_zero_eigenvalues` with
  a shift.

The four walker subtests that used to fail (`SingleDet.build` iterates
`hamiltonian.h1` per spin) pass since `from_file` sets a per-spin `h1`.

The walker tests skip when the LAFQMC walker modules are not importable,
and the HDF5 tests skip without h5py.

## Performance and memory

`tests/reference/benchmark.py --outer 20` times one relaxed evaluation at a
fixed gauge and 20 weak-Wolfe L-BFGS iterations. With `--cholesky-shift tied`
(or `split`) it uses the median Cholesky shift and optimizes it with the
gauge. On a 2-core container (23 November 2026):

| System | Relaxed evaluation | 20 L-BFGS iterations | Per outer evaluation: no shift / tied / split |
| --- | --- | --- | --- |
| H2/cc-pVDZ (ab_only, n=10, r=54) | 0.070 s | 2.1 s | 0.075 / 0.076 / 0.102 s |
| H4/6-31G (hybrid, n=8, r=32) | 0.307 s | 7.9 s | 0.305 / 0.313 / 0.316 s |
| LiH/STO-3G (hybrid, n=6, r=21) | 0.059 s | 1.4 s | 0.060 / 0.062 / 0.066 s |

The base commit of the Cholesky-shift work (`3290327`) measures the same on
this machine, with identical losses. The previous README's numbers (0.050,
0.224 and 0.044 s per relaxed evaluation) were measured in an earlier
session, so the difference is the machine, not the code. A tied shift costs
1–3% per outer evaluation (one diagonalization of `h_sigma` per trial); the
per-spin bases of a split shift cost 4–36%, most on ab_only H2.

The pair-loop implementation this package replaced was estimated at ~20 h
per relaxed evaluation for H2/cc-pVDZ. For hybrid systems the time is
dominated by `ModeSpace.orbital_K`. An optional patch from step 4
(`step4b_orbital_K.patch`) rewrites it as two GEMMs: H4 then took 0.038 s
per evaluation, against 0.224 s at the time.

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
  100). It no longer takes `eri`; it builds it from `chol`. Since roadmap
  step 13 it optimizes a tied Cholesky shift by default
  (`optimize_cholesky_shift="tied"`; `False` restores the previous behaviour).
- **Energies.** `parse_decomposition` no longer adds `mu.N` to `E0`.
  Λ is `ham.denom + mu.N` (+ `ham.cholesky_energy` with a Cholesky shift), and
  `from_file` leaves `E0 = 0`.
- **File format 3.** Per-spin one-body bases (`h1_X` is `(k, n, n)`), Cholesky
  shift attributes and `chol_lam`; `from_file` sets `ham.h1` per spin
  `(2, n, n)`. Version 2 files still load.
- **Cholesky shifts.** Decompositions and `GaugeEvaluator` accept a fixed
  `cholesky_shift`: forward model, inner solve, and the gauge and shift
  gradients (`GaugeGradient.cholesky_gradient`, option `cholesky_gradient=`),
  and the outer optimization of the shift
  (`optimize_gauge(optimize_cholesky_shift=...)`), saved in file format 3.
- **Outer points.** `ExponentialEvaluation` gains `coordinates` and
  `gradient`, which are what L-BFGS steps on; without a shift they are `theta`
  and `gradient_theta`. `near_zero_eigenvalues` returns
  `(factor, mode, lambda, spin)` for a shifted decomposition.
- **`check_gauge_regularity`** takes optional `eigenvalues=` (the shifted
  eigenvalues for the zero test) and `one_body_eigenvalues=`.
- **Documentation.** The PDF covers the Cholesky shift and file format 3, and
  is built from `docs/build_documentation.py` (the `docs/build/` sources the
  previous README named were never in the repository). `benchmark.py` gains
  `--cholesky-shift`.
