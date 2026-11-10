# Weighted loss and spin-resolved one-body shifts

Base: r-peng/ipie, branch lafqmc_thc, commit a91a66a2911dab3b6708189e1ee47becfe6e3462
("before implementing Lweighetd"), which already contains the weak-Wolfe change.

All defaults reproduce the previous behavior (`loss_kind="raw"`, zero shift,
`optimize_shift=False`). The 136 previous tests pass unchanged; 23 new tests
were added (159 total, 3 HDF5 tests skip without h5py).

## What is optimized

Parameters: Cholesky gauge theta (outer L-BFGS, unchanged), partition ratios t
(inner), and the spin-resolved shifts (mu_up, mu_down) (inner, optional). Weights
stay fixed: a_k = 1/tau1 for one-body channels; 1/tau2 per quartet channel and
1/(2*tau2) per two-branch channel, as before.

Objective (`loss_kind="weighted"`):

    L_w = sum_{i<j} a_i a_j ||[u_i, u_j]||_F^2,   a_i unnormalized.

It equals `commutator_loss(dec, normalize_weights=False).weighted_comm2`; the
normalized diagnostic is `L_w/denom**2`.

## Key identities used

1. `a_i a_j ||[A_i,A_j]||^2 = ||[sqrt(a_i) A_i, sqrt(a_j) A_j]||^2`, so `L_w` is
   `L_raw` on blocks whose MERGED amplitudes are scaled by `sqrt(a_i)`. All
   kernels, the exact per-source profile, the analytic t update, the KKT
   residual, the moving-bound correction and the eigensystem/rotation/exponential
   pullbacks are shared. Leg adjoints are chained back by `sqrt(a_i)`.
2. With fixed weights, `tau2` drops out of `L_w` at fixed t (both maps); it still
   sets the feasible t intervals.
3. Shift: `d[k,s] = (mu_s - eps_k)*tau1`, and
   `lambda_electronic = denom + mu_up*N_up + mu_down*N_down` reconstructs
   `lambda*I - H = sum_i a_i U_i` on the (N_up, N_down) sector. A nonzero shift
   requires `nelec`; `aa_only` has only mu_up.
4. One-body channels commute, so either loss is exactly quadratic in each mu_s
   and the spins decouple:
   `L(mu_s) = const + tau1^2 sum_k w_k (mu_s - eps_k)^2`,
   with minimizer `sum_k w_k eps_k / sum_k w_k` (a weighted mean of the orbital
   energies), clipped to `mu_s >= max(eps) - one_body_limit/tau1`.
   For two orbitals it is `mean(eps)` at every gauge and partition.

## Inner algorithm with `optimize_shift=True`

1. Set mu to mu*(t0) (no shift seed; the incoming shift only needs to be feasible).
2. Check the t residual; stop with zero sweeps if it already passes.
3. Each sweep: sequential exact t updates, then the exact mu step, then a full
   loss refresh and the residual check.

At every check mu is exactly optimal at the current t, so its gradient is zero
(or wall-filtered at the lower bound) and needs no threshold. The added shift
test is on the step: the last sweep must move each mu by at most
`shift_atol + shift_rtol*(max(eps) - min(eps))` (defaults 0 and 1e-8). When the t
residual is converged this is implied up to the linear coupling dmu*/dt; it is a
secondary safeguard. `relaxed_gauge_gradient` re-checks that the stored mu is
the exact minimizer at the stored t. mu's bound depends on h only, so the
envelope gauge gradient gets no extra moving-bound term.

## File changes

| File | Change |
| --- | --- |
| `models.py` | `LOSS_KINDS`; `OneBodyDecomposition.shift` (default zero) and `.spins`; `FixedDecomposition.shift`, `.shift_energy`, `.lambda_electronic`; new `ShiftCoefficients`; `PartitionSweep.shift_change`; `PartitionOptimizationResult` gains `loss_kind`, `optimize_shift`, `shift_bounds`, `shift_gradient`, `shift_tolerance`, `input_loss`, `one_body_limit`, `shift_atol`, `shift_rtol` and `.shift`; `InnerOptions` gains `optimize_shift`, `one_body_limit`, `shift_atol`, `shift_rtol`; `OptimizerIteration`/`OuterIteration` gain `shift`. All appended with defaults. |
| `channels.py` | `build_one_body_channels(problem, tau1, shift=None)`; `one_body_at_shift` reuses the cached h eigensystem; `shift_pair`; `build_fixed_decomposition(..., shift=None)`. |
| `loss.py` | `pair_scales`, `loss_kind_name`; `OverlapEvaluator(..., loss_kind=)` scales merged block amplitudes; `OverlapEvaluator.total()`; `objective_loss(dec, loss_kind=, method=)`. `evaluate()` stays raw-only. |
| `shift.py` (new) | `shift_bounds`, `shift_tolerance`, `shift_scale`, `with_shift`, `check_one_body_shift`, `shift_profiles`, `shift_coefficients`, `optimal_shift`. Layer order: models, problem, channels, loss, shift, partitions, gradients, optimizers. |
| `partitions.py` | Workspace takes `loss_kind`, scales own legs and rebuilt blocks, adds `shift_coefficients`/`update_shift`; `partition_coefficients(..., loss_kind=)`; `optimize_partitions(..., loss_kind=, optimize_shift=, one_body_limit=, shift_atol=, shift_rtol=)`. |
| `gradients.py` | `_loss_pullback(dec, loss_kind)`; `fixed_partition_gradient`/`normalized_partition_gradient(..., loss_kind=)`; `relaxed_gauge_gradient` uses the result's `loss_kind` and re-checks mu; `GaugeEvaluator(..., loss_kind=, shift=)` with per-call `shift` in `decomposition`/`fixed`/`fixed_z` and `optimize_shift` etc. in `relaxed`; one-body consistency is validated. |
| `optimizers.py` | Iteration records carry the relaxed shift; verbose lines append `mu=(...)`. L-BFGS, Armijo and weak Wolfe are unchanged. |
| `__init__.py` | Exports the new names. |
| `README.md` | New section "Weighted loss and spin-resolved shifts"; file and test lists. |
| `tests/test_weighted.py` (new) | 7 tests. |
| `tests/test_shift.py` (new) | 16 tests. |

## Observations

- `hybrid` channels are spin-symmetric when the diagonal-source ratios vanish.
  On the test fixture the relaxed diagonal ratios went to zero (|t| < 1e-15) from
  nonzero starts, so the relaxed mu_up = mu_down there. `ab_only` (left leg always up) breaks the symmetry and gives slightly
  different mu_up, mu_down.
- Independent of these changes, Gauss-Seidel partition sweeps can converge
  slowly (linearly, rate close to one) for `ab_only` with n=3: on the shared test
  fixture the raw, unshifted t residual was still about 7e-7 after 150 sweeps.
  `hybrid` fixtures converge in 10-20 sweeps. Tight inner tolerances on
  `ab_only` therefore need a large `max_sweeps` or an accelerated inner solver.
