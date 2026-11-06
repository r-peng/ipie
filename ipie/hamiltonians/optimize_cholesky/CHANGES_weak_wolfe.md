# Weak-Wolfe line search (fix A)

Base: r-peng/ipie, branch lafqmc_thc, commit e9e08ba869739387f3643a722e872eb33799dc49.

## What changed

| File | Change |
| --- | --- |
| `models.py` | `LBFGSOptions` and legacy `OuterOptions` gain `line_search` (`"armijo"` default, `"weak_wolfe"`), `wolfe_c2` (0.9) and `wolfe_max_step_norm` (pi), appended after the existing fields so positional layouts and `LBFGSOptions(**asdict(OuterOptions(...)))` still round-trip. `OuterIteration` gains `acceptance` (default `""`). |
| `optimizers.py` | `LBFGSOptimizer._step` now computes the direction, dispatches to `_armijo_search` (the original backtracking, moved verbatim) or the new `_weak_wolfe_search`, then applies the unchanged curvature-pair update. |
| `problem.py`, `__init__.py` | New public helper `near_zero_eigenvalues(state, rtol=1e-6, atol=0)` reporting active kinks. |
| `tests/test_weak_wolfe.py` | 11 new tests: options, smooth quadratic, axis-aligned and curved kinks, step expansion and cap, `armijo_only` fallback, invalid trials, failure and budget paths, helper, physical two-orbital descent. |
| `README.md` | New section "Weak-Wolfe line search for kinked objectives"; debugging table and test list updated. |

The default remains `line_search="armijo"`: existing scripts give identical results.
All 125 original tests and 11 new tests pass (136 total, including HDF5 tests when h5py is installed).

## Use

```python
options = OuterOptions(max_iterations=200, verbosity=2, line_search="weak_wolfe")
result = optimize_gauge(evaluator, theta0=theta, options=options, inner_options=...)
print(result.status, result.loss, near_zero_eigenvalues(result))
```

## Install

From the repository root, replace the directory rather than extracting over it:

```sh
cd ipie/hamiltonians
mv optimize_cholesky optimize_cholesky.bak     # or delete it
unzip /path/to/optimize_cholesky_weak_wolfe.zip
python -m unittest discover -s optimize_cholesky/tests -v   # or from the repo root as in README
```
