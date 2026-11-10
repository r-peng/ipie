# Cholesky gauge and partition optimization

This package builds an exact density-channel decomposition and minimizes its
squared commutator loss, either unweighted (`loss_kind="raw"`, default) or
weighted by the stored Green weights (`loss_kind="weighted"`). It supports
orthogonal Cholesky gauges, fixed or relaxed partitions, optional relaxed
spin-resolved one-body shifts (mu_up, mu_down), analytic gauge gradients, and
explicit L-BFGS.
NumPy derivatives and L-BFGS are implemented. PyTorch derivatives and GDM are
reserved options that raise `NotImplementedError`; neither silently falls back.

## Nine production files

| File | Responsibility and useful entry points |
| --- | --- |
| `models.py` | Data, options, exceptions and result records; `PreparedProblem`, `FixedDecomposition`, `GaugeGradient`, `OuterOptimizationResult` |
| `problem.py` | Input validation, HDF5 loading, sector selection, orthogonal gauges and eigensystems; `prepare_problem`, `load_problem`, `build_gauge_state` |
| `channels.py` | Source enumeration, amplitude maps, positive channel weights, rotation matrices; `build_fixed_decomposition` |
| `loss.py` | Dense and overlap commutator kernels for an existing decomposition; `raw_loss`, `objective_loss`, `commutator_loss`, `OverlapEvaluator` |
| `shift.py` | Spin-resolved one-body shifts: bounds, exact quadratic profiles, `with_shift`, `shift_coefficients`, `optimal_shift` |
| `partitions.py` | Bounds, source-local coefficients, sequential analytic partition updates; `partition_coefficients`, `optimize_partitions` |
| `gradients.py` | Derivative backend selection, elementary pullbacks, fixed/relaxed gauge derivatives and `GaugeEvaluator` |
| `optimizers.py` | Exponential objective, solver-neutral parent, explicit L-BFGS child, GDM placeholder and `optimize_gauge` |
| `__init__.py` | Stable package-level public imports; no numerical implementation |

Read the files in the table's order: models, problem, channels, loss,
shift, partitions, gradients, then optimizers. Each numerical layer builds on the
earlier layers. Solver-independent and solver-specific classes share a file
but retain separate responsibilities. No solver or backend subdirectories are
needed for the current implementation.

`models.py` remains separate because its records cross every layer and can
appear in saved objects. Their original module names and field layouts are
preserved by this reorganization. This is not a new serialization format or a
promise that arbitrary pickled engines from old module paths remain portable.

### Old module paths

Package-level public names are preserved. Prefer imports such as
`from ipie.hamiltonians.optimize_cholesky import GaugeEvaluator`.
Direct submodule imports must follow this mapping; there are no shim modules.

| Previous module | Current module |
| --- | --- |
| `models` | `models` |
| `prepare`, `validation` | `problem` |
| `channels` | `channels` |
| `objective` | `loss` |
| `partitions` | `partitions` |
| `derivatives`, `gradients` | `gradients` |
| `outer`, `solvers.base`, `solvers.lbfgs`, `solvers.gdm`, `solvers` | `optimizers` |

The fetched source snapshot imported `outer.py` and an outer-test helper that
were missing from its tree. Their previously delivered implementations and
physical outer tests have been restored into the consolidated modules/tests.

## Inputs and quick start

Install NumPy and SciPy; HDF5 loading additionally requires h5py. Use the package
inside an installed ipie checkout or run from the repository root.

| Input | Shape | Convention |
| --- | --- | --- |
| `h1` | `(n,n)` | Bare/original physical one-body matrix, without a second correction |
| `eri` | `(n,n,n,n)` | `eri[p,r,q,s] = (pr|qs)` |
| `chol` | `(r,n,n)` | Real symmetric factors, `eri = einsum('gpr,gqs->prqs', chol, chol)` |

The supplied Cholesky decomposition is assumed exact and checked against ERI.
Arrays must be real, finite and compatible; complex inputs are rejected.
No fitting, truncation, hidden symmetrization or mean-field shift is applied.
Preparation returns owned, read-only float64 copies.

```python
import numpy as np
from ipie.hamiltonians.optimize_cholesky import (
    prepare_problem, load_problem, ValidationOptions, GaugeEvaluator,
    InnerOptions, LBFGSOptions, optimize_gauge, raw_loss,
)

# h1, eri and chol have already been computed elsewhere.
problem = prepare_problem(
    h1, eri, chol, nelec=(1, 1),
    options=ValidationOptions(atol=1e-12, rtol=1e-10, block_size=16),
)
# Alternative: one .h5 file with datasets named h1, eri and chol.
# problem = load_problem("integrals.h5", nelec=(1, 1))

engine = GaugeEvaluator(
    problem, tau1=0.02, tau2=0.02,
    partition="balanced_log", backend="numpy",
)

def report(row):
    print(row.iteration, row.loss, row.gradient_norm,
          row.tangent_gradient_norm, row.inner_sweeps)

result = optimize_gauge(
    engine, solver="lbfgs",
    inner_options=InnerOptions(gradient_atol=1e-12, gradient_rtol=0),
    options=LBFGSOptions(max_iterations=100, memory=10),
    callback=report,
)
print(result.status, result.initial_loss, result.loss)
decomposition = result.decomposition
print(raw_loss(decomposition))  # same L_raw as result.loss
```

The illustrative scales are not universal defaults for your integrals.
Choose positive `tau1`, `tau2` so the amplitudes are feasible. One-body channels
require `1 - epsilon*tau1 > 0`; positive one-body amplitudes above one are
allowed. Paired two-body channels require `abs(d) < amplitude_bound <= 1`.
Smaller `tau2` can restore feasibility at more gauges. Amplitudes are checked,
not clipped to repair an invalid decomposition.

`load_problem` can use three files via `eri_path`, `chol_path`, with configurable
`h1_key`, `eri_key`, `chol_key` (including HDF5 group paths). It reads the full
arrays into memory and does not write to the input files.

## Channel, spin and partition contracts

Particle counts select the policy automatically. With no counts, use generic
`hybrid`. Up and down spins are encoded by 0 and 1. A down-only sector remains
generic; the implementation does not silently exchange spin labels.

| Policy | Selected sector | Sources per factor | Two-body channels per factor |
| --- | --- | --- | --- |
| `hybrid` | Generic cases | `n*(n+1)/2` | `2*n*n` |
| `ab_only` | Exactly `(N_up,N_down)=(1,1)` | `n*n` | `2*n*n` |
| `aa_only` | `N_down=0` | `n*(n-1)/2` | `n*(n-1)` |

Hybrid diagonal spatial sources have two opposite-spin branches. Its
off-diagonal spatial sources have a four-spin quartet. The reduced policies
use the two-branch identity for every retained spin-resolved product and are
Hamiltonian identities on the selected particle sector, not full-Fock-space
identities. All policies retain formal zero couplings and channel multiplicity.
One-body terms add `2*n` channels, or `n` for `aa_only`.

Each source owns one partition coordinate `t[i]`, shared by that source's
branches/quartet. It is not one parameter per emitted channel. For eigenvalues
`lL`, `lR`, define `W=lL*lR` and `k=abs(W*tau2)`:

| Map | `dL` | `dR` |
| --- | --- | --- |
| `balanced_log` | `sign(lL)*sqrt(k)*exp(t)` | `-sign(lR)*sqrt(k)*exp(-t)` |
| `anchored_log` | `lL*sqrt(tau2)*exp(t)` | `-lR*sqrt(tau2)*exp(-t)` |

Both give `dL*dR = -W*tau2`; the sign follows the positive-Green-weight
convention in this implementation. Balanced zero-coupling sources have both
legs zero; anchored sources can retain one nonzero leg when the other is zero.
For nonzero eigenvalues, the maps describe the same ratios with shifted `t`.

For a fixed-gauge reference decomposition:

```python
from ipie.hamiltonians.optimize_cholesky import (
    build_gauge_state, build_fixed_decomposition, commutator_loss,
    partition_coefficients, optimize_partitions,
)
gauge = build_gauge_state(problem.data)  # identity Cholesky gauge
fixed = build_fixed_decomposition(problem, gauge, tau1=.02, tau2=.02, t=0.)
diagnostics = commutator_loss(fixed, method="overlap")
coefficients = partition_coefficients(fixed)
inner = optimize_partitions(fixed)
```

`FixedDecomposition` stores one-body channels first, then two-body channels.
Every two-body `Channel` contains `bix`, `spins`, positive `a_i`, `modes`, and
the two amplitudes `d`. `two_body.source_slices` maps each source to its channel
range; `two_body.t` and `amplitudes` are source-indexed. Keep these together.
The returned objects are ready for a future SOR adapter; that adapter and a
dedicated disk serialization interface are not implemented here.

Stored `a_i` values are positive, unnormalized Green weights. In this code's
convention, with zero shift, `H=sum_i a_i*(I-U_i)` and `denom*I-H=sum_i a_i*U_i`,
with `denom=sum_i a_i` and no added constant energy. With a shift, use
`lambda_electronic=denom+sum_s mu_s*N_s` in place of `denom` (on the sector). They are the negatives of the
coefficients called `a_i` in the original note. Do not normalize the one-body
and two-body lists independently.

`raw_loss` sums squared single-particle Frobenius commutators over all `i<j`
and both spin blocks. It has no coefficient weights. Disjoint spin blocks
commute; a pair contributes through its shared active spins. Dense and overlap
methods implement the same loss. `commutator_loss` additionally returns weighted
diagnostics compatible with SOR; those diagnostics are not optimized.

## Derivative backend and inner relaxation

`GaugeEvaluator(..., backend="numpy")` selects the analytic NumPy/SciPy path.
`backend="pytorch"` is recognized but raises before engine initialization does
numerical work. Unknown names raise an error; importing the package does not
import PyTorch. Backend validation and derivative code now live together in
`gradients.py`, including commutator, symmetric-eigensystem and exponential
Fréchet pullbacks. Future differentiable implementations belong at these
entry points, not in the solver parent.

The engine supports `fixed(O,t=...)`, `fixed_z(O,z=...)`, and `relaxed(O,...)`.
The first holds physical log ratios fixed; the second holds normalized
interval coordinates fixed; the third solves the partition problem at that
gauge and returns the reduced gradient. These are different derivatives.

The inner solver updates sources sequentially using their analytic `A,B,C`
profiles and then checks a fresh full projected gradient. Its feasible intervals
include amplitude bounds and the finite `max_abs_t` cap. Relaxed gradients
include the response of gauge-dependent bounds through the fixed-domain
coordinate map. Omitting that response is generally wrong at active walls.

General inner solutions are converged local stationary solutions on a tracked
branch, not certified global minima. Degenerate eigenpairs, balanced zero
eigenvalues and certain bound switches have explicit regularity errors rather
than arbitrary eigenvalue-gap fixes. Tight outer accuracy requires sufficiently
tight inner tolerances. For the two-spatial-orbital `(1,1)` balanced setting,
zero ratios can satisfy inner stationarity immediately and give zero sweeps.

## Objective and solver classes

The exponential objective exposes
`F(theta)=L_raw(exp(S(theta))@reference, t_star(theta))`, with strictly upper
triangle angle packing `(0,1),(0,2),...,(1,2),...`. It re-solves partitions at
every new trial gauge, including rejected trials, then applies the exact
coordinate-gradient pullback. There is no differentiation through inner sweeps.

```python
from ipie.hamiltonians.optimize_cholesky import (
    ExponentialGaugeObjective, LBFGSOptimizer, LBFGSOptions,
)
objective = ExponentialGaugeObjective(engine, inner_options=InnerOptions())
solver = LBFGSOptimizer(objective, options=LBFGSOptions(memory=10))
result = solver.optimize()  # initial=None means zero theta
# Or: result = solver.optimize(theta0, callback=report)
```

`reference` is fixed for the entire objective. To start from `O_initial`, set
`reference=O_initial` and use zero theta. Reflections are allowed. The fixed
exponential solver does not recenter or transport history between charts.

`inner_t0` is copied and frozen. Every trial projects that same seed into its
own intervals; with `None`, each trial starts at its own interval midpoints.
Accepted/rejected partitions never become another trial's seed. A one-entry
cache holds matching theta, loss, gradient and inner snapshot. Do not mutate
the underlying engine configuration while using an objective.

`OptimizerOptions` contains common budgets, stopping and final validation.
`LBFGSOptions` adds memory and line-search settings. Legacy `OuterOptions`, its
positional field order, and existing default `optimize_gauge(...)` calls remain
supported. `solver="lbfgs"` is the default. `solver="gdm"` raises
`NotImplementedError` before objective construction; `GDMOptimizer` is also a
concrete placeholder whose `optimize` raises before evaluation. No geometric
GDM algorithm or Q-Chem-equivalent implementation is provided yet.

`OuterOptimizer` knows neither a solver name nor a coordinate representation.
Its `_initialize`, `_step`, `_build_result` hooks belong to children. It owns
common budgets, callbacks, stopping and accepted-state bookkeeping. A future
GDM child can use matrix-valued gauge points and geometric paths/transport.
Solver selection occurs in the convenience function, outside the parent.

The child contract is explicit:

| Hook | Required return value |
| --- | --- |
| `_initialize(initial)` | `(initial_point, fresh_solver_state)`; initial failures propagate |
| `_step(point, run, iteration)` | `OptimizerStep(evaluation=accepted_point, ...)`, or a failure status with no evaluation |
| `_build_result(point, summary)` | This solver's result containing the last accepted point and run metadata |

The objective supplies `evaluate(parameters)` and an `evaluations` counter.
Points supply `loss`, `gradient_norm`, `tangent_gradient_norm`, and
`gauge_gradient.inner_result` for partition diagnostics. The parent does not
require `theta` or `reference`. A child performing several trial evaluations
must respect `_budget_exhausted(run)` inside its step. With final validation
enabled, the objective supplies `validate(point)` and the result is a dataclass
with a `validation_report` field. Add a future solver's class here and its
options in `models.py`; implement its selection in `optimize_gauge` without
changing the parent.

Calling the same solver again starts fresh curvature memory and a fresh budget.
Objective caches may persist. Result counters are per run; a cached initial
point can report zero fresh evaluations and one cache hit. Objective counters
remain cumulative.

## Debugging explicit L-BFGS

All algorithm steps are visible in `optimizers.py`; no SciPy optimizer hides
the iteration. Start with these functions/methods:

| Location | What to inspect |
| --- | --- |
| `_lbfgs_direction` | Two-loop inverse-Hessian product and last-pair scaling |
| `LBFGSOptimizer._step` | Descent fallback, displacement cap, search dispatch, curvature acceptance |
| `LBFGSOptimizer._armijo_search` | Original sufficient-decrease backtracking (default) |
| `LBFGSOptimizer._weak_wolfe_search` | Lewis-Overton weak-Wolfe bracketing for kinked objectives |
| `ExponentialGaugeObjective.evaluate` | Gauge, inner convergence, coordinate pullback, joint cache and work counters |
| `OuterOptimizer.optimize` | Accepted-state lifecycle, budgets, callbacks and stopping |

Without history, steepest descent is normalized. A nonfinite/non-descent
quasi-Newton direction clears memory and falls back to steepest descent.
`max_step_norm` caps the initial trial displacement; Armijo accepts only
`F(theta+alpha*p) <= F(theta)+armijo*alpha*(gradient_theta@p)`.
With the default `line_search="armijo"`, backtracking multiplies alpha by
`backtrack`; this is not a Wolfe search. See the next section for
`line_search="weak_wolfe"`.
Curvature pairs use `s=theta_new-theta`, `y=g_new-g`; only sufficiently positive
`s@y` values are retained, with at most `memory` pairs.

Feasibility, regularity, inner-convergence and recoverable numerical trial
failures trigger backtracking. Programming/configuration exceptions propagate.
An invalid initial evaluation raises. A failed line search or exhausted budget
returns the last accepted snapshot, never a rejected trial or a fake gradient.

`result.history` and callbacks contain initial/accepted scalar records.
`result.trials` records every attempted line-search point and rejection reason;
`attempt` starts at one in each iteration. Inspect `curvature_update`,
`memory_size`, `line_search_trials`, `inner_projected_gradient_norm`, and:

```python
for trial in result.trials[-5:]:
    print(trial.iteration, trial.attempt, trial.step_size, trial.reason)
```

Convergence requires both the theta-gradient infinity norm and physical
tangent-gradient Frobenius norm to meet `atol+rtol*initial_norm`. A small
coordinate gradient with a nonzero physical residual gives `coordinate_stalled`.
Other statuses include `max_iterations`, `max_evaluations`, `line_search_failed`,
`stalled`, and `callback_stopped`. Raising `StopIteration` from a callback stops
cleanly at its accepted state. Small steps or loss changes alone do not certify
convergence; nonsmooth transitions can cause a nonconverged exit after descent.

`result.loss` and `initial_loss` both use relaxed partitions at their respective
gauges. `result.decomposition`, `O`, `theta`, and `partition_result` describe one
matching accepted state. Full gauge/ERI validation runs once outside the loops
by default and is stored in `validation_report`; use `validate_final=False` to
omit it. No outer or inner trial reconstructs ERI for this final reference check.

## Weak-Wolfe line search for kinked objectives

The relaxed reduced loss is not smooth. For one source the inner problem is
`min_t C + A*exp(2t) + B*exp(-2t) = C + 2*sqrt(A*B)` (AM-GM), and with the
anchored amplitudes `A ~ ell_alpha**2`, `B ~ ell_beta**2`. The square root makes
the reduced loss depend on `|ell_alpha*ell_beta|`, so it has a kink on every
surface where an eigenvalue of a rotated factor crosses zero (the inner
minimizer escapes to `t = +-infinity` there, so Danskin's theorem fails). The
balanced map shows the same `|ell|` already at fixed `t`. For two orbitals with
`nelec=(1,1)` the reduced loss is exactly
`4*tau2*sum_g H_g*nuc_g**2 + 8*tau2**2*sum_{g<e} K_ge*nuc_g**2*nuc_e**2`, with
`nuc_g` the nuclear norm (sum of absolute eigenvalues) of rotated factor `g`.
Nuclear norms reward rank-deficient factors, so minimizers typically lie ON
kinks, where the gradient does not vanish and gradient tests cannot pass.

`line_search="weak_wolfe"` (in `LBFGSOptions` or legacy `OuterOptions`) uses the
Lewis-Overton bracketing search intended for such objectives. A trial is
accepted when

    F(theta+alpha*p) <= F(theta) + armijo*alpha*(g@p)          (decrease)
    g(theta+alpha*p)@p >= wolfe_c2*(g@p)                        (weak curvature)

A decrease failure, or a recoverable invalid trial, becomes the upper bracket; a
curvature failure becomes the lower bracket. A finite bracket is bisected;
otherwise alpha doubles, up to the displacement cap `wolfe_max_step_norm`
(default pi radians in theta). The curvature test must be the weak, one-sided
one: at a kink the directional derivative jumps, so strong Wolfe has no
solution, whereas a point just beyond the kink satisfies the weak test. Every
accepted Wolfe step therefore gives `s@y > 0`, the curvature pair is kept, and
the quasi-Newton model learns the large curvature across the kink while still
moving along the kink surface.

If no trial satisfies both conditions within `max_line_search` attempts, the
lowest-loss trial that satisfied the decrease condition is accepted, labelled
`armijo_only` (or `expansion_capped` if expansion stopped at the cap). Only if
no trial decreased the loss does the step return `line_search_failed`; at a
nonsmooth minimizer this is the expected termination, not a programming
failure. Each accepted record carries `acceptance` (`armijo`, `weak_wolfe`,
`armijo_only`, `expansion_capped`); trial reasons add `insufficient_curvature`.
With `verbosity=2` each trial prints its `slope_ratio=g(theta+alpha*p)@p/(g@p)`,
which must be `<= wolfe_c2` for the curvature test to pass.

```python
from ipie.hamiltonians.optimize_cholesky import near_zero_eigenvalues
result = optimize_gauge(engine, theta0=theta,
                        options=OuterOptions(max_iterations=200, line_search="weak_wolfe"),
                        inner_options=InnerOptions(max_sweeps=500, gradient_atol=1e-12,
                                                   gradient_rtol=0))
print(result.status, result.loss)
print(near_zero_eigenvalues(result))   # ((factor, mode, eigenvalue), ...) = active kinks
```

`near_zero_eigenvalues(state, rtol=1e-6, atol=0)` accepts a `GaugeState`, a
`FixedDecomposition`, or any result with `.decomposition`, and reports
eigenvalues with `|ell| <= atol + rtol*max|ell|` within each factor.

Options: `line_search` (`"armijo"` default, or `"weak_wolfe"`), `wolfe_c2`
(default 0.9, requires `armijo < wolfe_c2 < 1`), `wolfe_max_step_norm` (default
pi, must be at least `max_step_norm`). The first trial keeps the existing
`initial_step` and `max_step_norm` behavior. The default remains `armijo`, so
existing scripts and results are unchanged unless `weak_wolfe` is requested.

On H2/STO-3G (Lowdin orbitals, R=2.4 bohr, r=3, tau1=tau2=0.05, balanced map),
from six random starts with |theta| <= 0.1 Armijo stopped at reduced losses
between 1.40401e-3 and 1.41642e-3; weak Wolfe reached 1.4034834e-3 from four
starts, 1.4034836e-3 from a fifth, and the same 1.41642e-3 as Armijo from the
sixth, which is a distinct local minimum on another pair of kinks. Weak Wolfe
used 1.4-3.9 times as many evaluations, because it keeps making progress
along the kink surface instead of stopping early. It does not choose basins:
use several starting gauges when local minima matter.

## Weighted loss and spin-resolved shifts

`loss_kind="weighted"` optimizes
`L_w = sum_{i<j} a_i*a_j*||[u_i,u_j]||_F^2` with the stored, UNNORMALIZED
`a_i` (equal to `commutator_loss(..., normalize_weights=False).weighted_comm2`;
the normalized diagnostic differs by the constant `1/denom**2`). Because
`a_i*a_j*||[A_i,A_j]||^2 = ||[sqrt(a_i)*A_i, sqrt(a_j)*A_j]||^2`, the weighted
loss is the raw kernel on blocks whose merged amplitudes are scaled by
`sqrt(a_i)`; every kernel, the exact `C+A*exp(2t)+B*exp(-2t)` source profile,
the analytic partition update and all gauge pullbacks are shared. With fixed
`a_k=1/tau1` and two-body weights, `tau2` drops out of `L_w` at fixed `t`
(it still sets the feasible intervals).

The shift writes the one-body term as
`sum_{k,s}(eps_k-mu_s) n_{k,s} + sum_s mu_s N_s`, so `d[k,s]=(mu_s-eps_k)*tau1`
and `FixedDecomposition.lambda_electronic = denom + mu_up*N_up + mu_down*N_down`
(add `E0` for `Lambda_total`). A nonzero shift therefore requires `nelec`.
For `aa_only` only `mu_up` exists. One-body channels commute, so either loss
is exactly quadratic in each `mu_s` and the spins decouple; the minimizer is
the weighted mean `sum_k w_k*eps_k/sum_k w_k`, clipped to the one-sided bound
`mu_s >= max(eps) - one_body_limit/tau1` (default `one_body_limit=1-1e-8`).
For two orbitals it equals `mean(eps)` at every gauge and partition.

```python
engine = GaugeEvaluator(problem, tau1=0.03, tau2=0.04, loss_kind="weighted")
result = optimize_gauge(
    engine, options=LBFGSOptions(line_search="weak_wolfe", verbosity=1),
    inner_options=InnerOptions(optimize_shift=True, gradient_atol=1e-12, gradient_rtol=0),
)
print(result.loss, result.decomposition.shift, result.decomposition.lambda_electronic)
```

With `optimize_shift=True` the inner solve sets `mu` to `mu*(t0)` before the
first residual check (no shift seed exists; the engine's `shift` only needs
to be feasible), then every sweep updates all `t` sequentially and then each
`mu` exactly. At every convergence check `mu` is therefore the exact minimizer
at the current `t`, its gradient is zero (or wall-filtered), and no gradient
threshold is needed for it. The added shift test requires the last sweep to
have moved each `mu` by at most `shift_atol + shift_rtol*(max(eps)-min(eps))`
(defaults `0` and `1e-8`). The relaxed gauge gradient re-checks that the stored
`mu` is the exact minimizer; `mu`'s bound depends on `h` only, so the envelope
gradient has no moving-bound term for it. Fixed shifts are passed as
`GaugeEvaluator(..., shift=(mu_up, mu_down))` or per call (`fixed`, `fixed_z`,
`decomposition`).

`hybrid` channels stay spin-symmetric when diagonal-source ratios are zero,
and then `mu_up=mu_down`; `ab_only` (left leg always up) and nonzero diagonal
ratios break the symmetry. Verbose outer output appends `mu=(...)` when the
shift is relaxed, and `OuterIteration.shift` records it.

## Tests and installation

Nine test modules group the same responsibilities as the implementation:
`test_problem` combines input/model checks; `test_channels` checks reconstruction;
`test_loss` compares dense/overlap/SOR losses; `test_partitions` checks analytic
updates; `test_gradients` includes backend and moving-bound checks; and
`test_optimizers` combines physical, controlled-quadratic, class and dispatch
tests; and `test_weak_wolfe` checks the weak-Wolfe search on smooth, kinked
and physical objectives, plus `near_zero_eigenvalues`; `test_weighted` checks
the weighted objective against explicit dense weights, its exact profiles,
inner solve and gradients; and `test_shift` checks shifted channels, Lambda and
sector reconstruction, exact quadratic profiles, relaxation order and seed
independence, the lower bound, and relaxed gradients. These use small exact
systems and independent numerical references.

```sh
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_problem -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_channels -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_loss -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_partitions -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_gradients -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_optimizers -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_weak_wolfe -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_weighted -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_shift -v
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
```

HDF5 round-trip tests skip when h5py is unavailable. See the ZIP's root
[REORGANIZATION_INSTALL.md](../../../REORGANIZATION_INSTALL.md) for replacement/unzip instructions and the
exact source snapshot. Replacing the old directory removes obsolete module
files; extracting over it alone would leave them behind.
