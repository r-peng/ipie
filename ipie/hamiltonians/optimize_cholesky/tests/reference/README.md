# Frozen reference for the optimize_cholesky refactor (step 1)

Everything here pins the behaviour of `optimize_cholesky` at commit
`3680f2920370dd114c05503045c178ca54b632d0` (branch `lafqmc_thc`, 2026-11-20).
Later refactoring phases are accepted only if they reproduce these numbers to a
relative 1e-12.

Unzip at the repository root. The bundle only adds new files:

```
ipie/hamiltonians/optimize_cholesky/tests/test_reference_golden.py
ipie/hamiltonians/optimize_cholesky/tests/reference/
    README.md              this file
    reference_kernels.py   frozen oracle (do not edit)
    systems.py             synthetic fixture generator + systems.npz reader
    systems.npz            molecular integrals (written by make_systems.py)
    make_systems.py        regenerates systems.npz (needs PySCF)
    golden_3680f29.npz     golden numbers + JSON metadata
    record_golden.py       how golden_3680f29.npz was produced
    benchmark.py           timing script for every phase
```

Suggested tag for the frozen commit: `git tag optimize-cholesky-frozen 3680f29`.

## The oracle (`reference_kernels.py`)

A self-contained copy of the 3680f29 code paths behind
`GaugeEvaluator.fixed / fixed_z / relaxed`, `loss.objective_loss`, the inner
Gauss–Seidel solve with the shift, and the exponential-gauge VJP. The
arithmetic and the operation order are the package's. Validation and the
dataclass object model are removed, and it imports nothing from the package.
It therefore keeps working when the refactor removes the pair loops and
`Channel` objects. It works on plain arrays:

```python
oracle = ReferenceOracle(h1, chol, nelec, tau1=0.03, tau2=0.04)   # partition, loss_kind, shift optional
oracle.fixed(O, t=t)                         # dict: loss, gradient_O, partition_gradient, ...
oracle.relaxed(O, inner=InnerSettings(optimize_shift=True))
```

The one addition to the copied code is `loss_pullback(..., rows=(i0, i1))`.
It restricts the outer pair index so that a very large evaluation can be
split across processes; the default is the complete loop.

Before recording, the oracle was checked against the untouched package on all
24 synthetic configurations and the small molecules: fixed, fixed_z,
objective, and relaxed with and without the shift. Results are in
"Verification" below.

## Golden data (`golden_3680f29.npz`)

Keys are `<case>/<evaluation>/<field>`. Inputs (`input/O`, `input/t`,
`input/z`, `input/shift`) are stored next to the outputs, so every evaluation
can be replayed exactly. `metadata` is a JSON string with the frozen commit,
the commit the recording ran at (must be the same), library versions, τ₁, τ₂,
inner and outer settings, and one entry per case.

| Cases | Policy | Evaluations | Produced by |
|---|---|---|---|
| 24 synthetic (`syn_*`): n,r ∈ {(3,2), (4,3)} × (2,1)/(1,1)/(2,0) × balanced/anchored × weighted/raw | all three | fixed, fixed_z, objective (raw and weighted), relaxed with fixed shift, relaxed with relaxed shift | package |
| `h2_631g` (1,1) | ab_only | fixed, objective, relaxed (shift relaxed), 5-iteration L-BFGS | package |
| `h2_631g_triplet` (2,0) | aa_only | same | package |
| `h4_sto3g` (2,2) | hybrid | same | package |
| `lih_sto3g` (2,2) | hybrid | fixed, objective | package |
| `h2_ccpvdz` (1,1), n=10, r=54 | ab_only | fixed-t loss and gauge gradient | oracle |

The settings are τ₁ = 0.03 and τ₂ = 0.04, inner `max_sweeps=3000`,
`gradient_atol=1e-12`, `gradient_rtol=1e-10`. The outer runs use 5 L-BFGS
iterations, weak-Wolfe, `validate_final=False`. Gauges are
`exponential_gauge(theta)` with deterministic θ. Fixed t and z are drawn
inside the feasible intervals. H2/cc-pVDZ uses t = 0 and μ = (0.25, 0.25).

H2/cc-pVDZ is oracle-only because one fixed-t evaluation with pair loops takes
about 1.5 CPU-hours, in the package as well as in the oracle. Relaxed
evaluations at that size would take many hours, so the relaxed and outer
checks use the smaller systems.

## Tests (`tests/test_reference_golden.py`)

```
python -m pytest ipie/hamiltonians/optimize_cholesky/tests/test_reference_golden.py -m ""
```

(`-m ""` is needed because `pytest.ini` only selects marked tests.)

- `PackageGoldenTests` replays every case through the public entry points:
  `prepare_problem`, `GaugeEvaluator.fixed/fixed_z/relaxed`,
  `loss.objective_loss` and `optimize_gauge`. If `GaugeEvaluator` has a
  `kernel=` option, every kernel listed in `gradients.KERNELS` is checked.
- `OracleGoldenTests` confirms that the oracle still reproduces the golden file.
- `GoldenFileTests` checks the provenance recorded in the metadata.

Tolerance (decision 4): `max|a-b| <= 1e-12 * max|reference|`. For t and μ the
scale is `max(1, |reference|)`. Converged flags, outer iteration counts and
statuses must match exactly. Sweep counts and timings are not compared.

Large cases (H2/cc-pVDZ and LiH/STO-3G) are skipped for pair-loop kernels
(the package's `reference` kernel or the oracle) unless
`OPTIMIZE_CHOLESKY_GOLDEN_LARGE=1`. A fast kernel (`kernel="vectorized"`)
always runs them. Once production has
only fast kernels, set `PAIR_LOOP_KERNELS = ()` in the test module.

## Benchmark (`benchmark.py`)

```
python ipie/hamiltonians/optimize_cholesky/tests/reference/benchmark.py --outer 20 --json bench.json
```

The script times `prepare_problem`, one relaxed evaluation (the median of 3)
and, with `--outer`, an L-BFGS run. It does this on H2/cc-pVDZ (ab_only),
H4/6-31G and LiH/STO-3G (hybrid) for every available kernel. Pair-loop kernels
whose estimated relaxed time exceeds `--budget` (600 s) are reported with
the estimate only. `--json` appends results, so phases can be compared.

## Verification (done before shipping)

All checks ran on numpy 2.5.3 / scipy 1.18.1. Other BLAS builds can differ
at the 1e-15 level, far below the 1e-12 tolerance.

- **Oracle against the untouched package.** Maximum difference 0.0 (bitwise)
  on all 24 synthetic configurations: fixed, fixed_z, objective, and relaxed
  with and without the shift. The same holds for `h2_631g`,
  `h2_631g_triplet` and `h4_sto3g`, with identical inner sweep counts.
- **`test_reference_golden.py` on a clean 3680f29 checkout.** Every test
  passes. This took about 38 min on 2 cores, almost all of it pair loops, with
  the large cases skipped.
- **The earlier `vectorized_kernels.patch` (`kernel="vectorized"`) against
  this file.**
  - Every field is within 1e-12. Single evaluations are within 3e-14.
  - The 5-iteration L-BFGS trajectories are within 5e-13. The worst field is
    `outer/gradient_theta`, so trajectories have the least headroom.
  - At H2/cc-pVDZ, against the oracle: loss 1.7e-16, gradient 1.4e-14.

## Baseline timings

Measured with `benchmark.py --outer 20` in a 2-core container.

| System | Channels | Pair-loop relaxed evaluation | Vectorized patch: relaxed | Vectorized patch: 20 L-BFGS iterations |
|---|---|---|---|---|
| `h2_ccpvdz` (ab_only) | 10,820 | ~20 h (estimate); one fixed-t evaluation measured at 2.1 CPU-h | 0.42 s | 11.8 s (28 evaluations) |
| `h4_631g` (hybrid) | 4,112 | ~3 h (estimate) | 0.37 s | 9.3 s |
| `lih_sto3g` (hybrid) | 1,524 | ~24 min (estimate) | 0.09 s | 2.3 s |

The estimates come from the benchmark's pair-count model. They are rough, but
they are consistent with the measured H2/cc-pVDZ fixed-t time.

## Regenerating

Do not regenerate the golden file during the refactor; it is the yardstick.
`record_golden.py` refuses to run unless `HEAD` is the frozen commit. Its
docstring lists the exact commands, including the chunked H2/cc-pVDZ
evaluation.
