# Cholesky optimization: steps 1 through 6

This package implements input models, HDF5 loading, orthogonal Cholesky gauges,
factor diagonalization, deterministic identity checks, density-channel
construction, dense/overlap commutator loss, partition optimization, and
analytic gauge gradients, and fixed-exponential outer gauge optimization.
The outer driver has a solver-independent parent and a working L-BFGS child.
The `sor_base.py` adapter remains a future step.

The package lives at `ipie/hamiltonians/optimize_cholesky/`. Run the examples
and tests from the ipie repository root, or with ipie installed. Imports start
with `ipie.hamiltonians.optimize_cholesky`.
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
from ipie.hamiltonians.optimize_cholesky import (
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
HDF5 datasets. Preparation records the policy; step-2 channel builders use that policy to
generate the appropriate spin products. The original h remains unchanged.

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

## Channel contract

`Channel(bix, spins, a_i, modes, d)` has exactly the requested fields:

```python
from ipie.hamiltonians.optimize_cholesky import Channel
channel = Channel(
    bix=0, spins=(0, 1), a_i=5.0, modes=(2, 1), d=(0.1, -0.2)
)
```

`bix` indexes the **final** Cholesky eigenbasis, 0=up and 1=down; `d` contains
actual branch amplitudes, and `a_i` is the positive **unnormalized** coefficient
in Lambda*I-H. All four spin orientations are representable. A future adapter
can canonicalize `(1,0)` to `(0,1)` by swapping both legs, since opposite spins
commute. The step-2 builders enforce source enumeration, paired-domain constraints,
and branch sharing. The loader and gauge builder themselves still emit no
channels; call the channel builders described below.

The `GaugeState` object is the completed output of step 1. It holds owned
read-only arrays and diagnostics, with no optimizer backend or open HDF5
handle. A final optimized/channel result and its persistence will be added
later; the present state does not imply optimization has occurred.

## Run tests

From the ipie repository root:

```sh
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
```

The tests use independent dense contractions and temporary HDF5 files,
including custom dataset paths and separate input files.
When h5py is missing, the three actual HDF5 tests are explicitly skipped;
they execute normally after installing h5py. The array-only functions do not
import h5py. The present implementation is a NumPy/SciPy reference; a later
automatic-differentiation implementation must preserve these conventions
without passing differentiable tensors through NumPy conversions.

## Step 2: fixed partitions and density channels

```python
import numpy as np
from ipie.hamiltonians.optimize_cholesky import (
    load_problem, build_gauge_state, enumerate_sources,
    build_fixed_decomposition, build_two_body_channels, get_rotation_matrix,
)

problem = load_problem("integrals.h5", nelec=(1, 1))
gauge = build_gauge_state(problem.data)  # or use a step-1 rotated gauge
sources = enumerate_sources(problem)
t = np.zeros(len(sources))
decomposition = build_fixed_decomposition(
    problem, gauge, tau1=0.01, tau2=0.01, t=t,
    partition="balanced_log",
)

# Keep these two lists separate for the future sor_base adapter.
two_body_channels = decomposition.two_body.channels
one_body_channels = decomposition.one_body.channels
print(two_body_channels[:2])  # bix, spins, a_i, modes, actual signed d
print(decomposition.denom)    # sum of all UNNORMALIZED coefficients
U_up, U_down = get_rotation_matrix(decomposition, 0)
```

Choose the fixed positive scales small enough that all amplitudes are admissible.
There is no automatic rescaling or clipping. If only the two-body list is wanted,
call `build_two_body_channels(problem, gauge, tau2, t=...)`.
One-body channels can be built separately with
`build_one_body_channels(problem, tau1)`.

### Sources and spin policies

One entry in `sources` owns one ratio. Its two Eq. (10) branches or its four
Eq. (14) hybrid branches share that ratio. `source_slices[i]` gives the
half-open interval of **two-body** channels emitted by source i.
The order is fixed, including exactly zero couplings:

| Policy | Source order within each bix | N_source | N_two_body | Sum of two-body weights |
| --- | --- | --- | --- | --- |
| `hybrid` | alpha<=beta; diagonal UD two-branch, off-diagonal quartet | r*n*(n+1)/2 | 2*r*n*n | r*(2*n*n-n)/tau2 |
| `ab_only` | all ordered spatial (alpha,beta), always spins=(0,1) | r*n*n | 2*r*n*n | r*n*n/tau2 |
| `aa_only` | alpha<beta, spins=(0,0) | r*n*(n-1)/2 | r*n*(n-1) | r*n*(n-1)/(2*tau2) |

Hybrid quartets are emitted as (UU,UD,DU,DD), with common leg signs
(+,-,-,+), and each channel weight 1/tau2. Every Eq. (10) product emits the
+ branch followed by the - branch, each with weight 1/(2*tau2). Quartet
amplitudes cannot be assigned independent spin-channel ratios under this
identity. Different two-branch sources may have independent ratios.

`hybrid` is a full-Fock-space identity; `ab_only` is used only for (1,1),
and `aa_only` only for N_down=0. The builders reject a policy inconsistent
with the PreparedProblem particle counts. If particle counts are absent,
the generic hybrid policy is used. Fully spin-up systems with <=1 electron
still retain the formal source list; its two-body action vanishes on that
sector. No topology changes are made based on the sector or coupling size.

### Two partition maps

The finite scalar or vector `t` is a **fixed log ratio**, not a bounded z
coordinate and not an optimizer input. A scalar broadcasts to all sources;
`None` means zero. Let ell_L,ell_R be the source's two Cholesky eigenvalues.

- `balanced_log` (default): with k=abs(ell_L*ell_R*tau2),
  d_L=sign(ell_L)*sqrt(k)*exp(t),
  d_R=-sign(ell_R)*sqrt(k)*exp(-t).
  If either eigenvalue is zero, both amplitudes are defined to be zero.
- `anchored_log`: d_L=ell_L*sqrt(tau2)*exp(t),
  d_R=-ell_R*sqrt(tau2)*exp(-t).
  At zero coupling one leg may survive. The complete source still cancels
  correctly; these channels are not silently replaced by identities.

Both maps enforce d_L*d_R=-W*tau2 and use analytic constant weights even at
W=0. Both signs of each leg occur, so require abs(d)<amplitude_bound<=1.
The default bound is 1; supply a smaller positive bound for a safety margin.
Large fixed ratios or scales that violate the open domain raise ValueError.
Amplitudes are evaluated in logarithmic form to avoid gratuitous overflow.

`TwoBodyDecomposition` holds the gauge, policy, particle counts, scales,
source table, fixed ratios, base amplitudes, source slices, and channel tuple.
These metadata preserve the ownership needed by the future partition solver.
No optimizer or automatic differentiation is implemented at this step.

### One-body terms and normalization

The one-body matrix is the original physical h, which is jointly diagonalized
once. One channel per mode and retained spin has d=-epsilon*tau1 and weight
1/tau1. Both spins are retained for hybrid/ab_only; only up spin for aa_only.
Zero one-body eigenvalues are retained as identity channels. For these single
branches the actual real-exponential condition is d>-1; positive d>=1 is valid.

`FixedDecomposition` contains `one_body` and `two_body`. Its combined order is
one-body channels followed by two-body channels. `coefficients` is the raw
positive array; `denom` is their common sum B. The matrix helpers return
[u_up,u_down], with None for an inactive identity block, matching sor_test.py.
They multiply the stored rank-one factors literally, including any repeated
same-spin cross term in a manually constructed channel.

For E0=0, H=B*I-sum_i a_i*U_i. For an external scalar E0, Lambda=E0+B and
Lambda*I-H=sum_i a_i*U_i. Divide the **combined** raw sum by B when checking
the normalized Green function. Do not normalize the two lists separately.
Full two-body and one-body reconstruction is exact in the policy's stated
space, up to floating-point error and the input-validation tolerance.

### Step-2 reference tests

```sh
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
python -m ipie.hamiltonians.optimize_cholesky.tests.test_channels
```

The second command prints readable H/G/Slater residuals and runs the new tests.
Like `ipie/hamiltonians/tests/sor_test.py`, the tests use interleaved bitstrings,
`string_act` fermionic signs, determinant minors to lift rotations, and Slater
determinant coefficients. The reference Hamiltonian is independently built
from **bare h plus the canonical quartic ERI operator**; no extra normal-order
correction is subtracted in that reference. Tests check H, shifted raw G,
normalized G, individual density-factor actions, and action on a general
Slater determinant. Hybrid is also tested on full Fock space. Reduced policies
are tested in their physical sectors and explicitly shown to fail outside them.

Additional cases cover both partition maps, both signs of W, shared quartet
ratios, known channel/weight counts, rotated gauges, diagonal opposite spins,
zero couplings, a valid one-body d>1, empty two-body lists, and invalid domains.
Their numerical checks use NumPy/SciPy and the existing pure-Python
bitstring_utils. They do not call HDF5 I/O, PySCF, or sor_base construction.
Importing the full ipie package may still require its installed runtime
dependencies. The test module adds no new dependency beyond step 1.

## Step 3: reference commutator loss

```python
from ipie.hamiltonians.optimize_cholesky import (
    load_problem, build_gauge_state, build_fixed_decomposition,
    commutator_loss, raw_loss, OverlapEvaluator,
)

problem = load_problem("integrals.h5", nelec=(1, 1))
gauge = build_gauge_state(problem.data)
decomposition = build_fixed_decomposition(problem, gauge, .01, .01, t=0.0)

dense = commutator_loss(decomposition, method="dense")
fast = commutator_loss(decomposition, method="overlap")
print(dense.comm2, fast.comm2)
print(fast.as_dict())
loss = raw_loss(decomposition)  # unweighted squared objective only

# Reuse a fixed-state cache for pair-by-pair inspection.
evaluator = OverlapEvaluator(decomposition)
pair_squared_norm = evaluator.pair_norm2(0, 1)
same_total = evaluator.evaluate()
```

`commutator_loss` defaults to `method="overlap"`. Convenience functions
`dense_commutator_loss` and `overlap_commutator_loss` return the same result
type. All methods include both the one-body and two-body channel lists in
their existing combined order. They evaluate real float64 single-particle
matrices, not many-body Fock-space commutators.

### Exact agreement with compute_total_commutativity

For each unordered pair i<j, define

```
pair_ij = ||[u_i_up,u_j_up]||_F**2 + ||[u_i_down,u_j_down]||_F**2
```

`None` means identity in an inactive spin block. Squared spin norms are
added; there is no square root. The returned `CommutatorLoss` fields map to
`sor_base.compute_total_commutativity` as follows:

| Returned field | Existing printed diagnostic |
| --- | --- |
| `comm2` | `comm2` = sum(pair_ij) = L_raw |
| `weighted_comm2` | sum(a[i]*a[j]*pair_ij) |
| `weight` | sum(a[i]*a[j]) over **all** i<j |
| `weighted_mean` | `weighted_comm2/weight` |
| `pair_count` | nterms*(nterms-1)/2, including commuting/identity pairs |
| `weights_normalized` | Whether a[i] is the raw coefficient divided by denom |

By default, `normalize_weights=True`: a[i]=coefficient[i]/decomposition.denom.
This matches a parsed SumOfRotationBase object. To compare with raw coefficients,
pass `normalize_weights=False`. A common coefficient rescaling leaves comm2
and weighted_mean unchanged; it rescales the raw weighted numerator and
denominator by the square of that factor. The unweighted comm2 alone is the
optimization objective; weighted fields are diagnostics.

The comparison requires the same actual channels, partitions, and coefficient
convention. Fock-space equality of two Hamiltonian decompositions does not
guarantee equal L_raw. Identity channels contribute zero commutators but still
contribute to the weighted denominator. Retained anchored zero-coupling sources
can have nonidentity channels, even though their Hamiltonian source sum is zero.
No channels are coalesced or deleted by the objective evaluator.

For fewer than two channels the pair sum is empty: comm2=weighted_comm2=weight=0
and weighted_mean=None. This explicitly represents an undefined weighted mean;
the existing print-only sor_base function divides by zero in this case.

### Overlap formula and spin blocks

Within each spin sector write the channel increment as A=V D V.T, where the
active columns of V belong to one orthonormal h or Cholesky eigenbasis.
Each generated block has at most two active columns. For another increment
B=Z E Z.T and C=V.T @ Z, the public kernel `commutator_norm2(d,e,C)` uses

```
||[A,B]||_F**2 = 2*Tr(D**2 @ C @ E**2 @ C.T)
                 - 2*Tr((D @ C @ E @ C.T) @ (D @ C @ E @ C.T))
```

Here D=diag(d), E=diag(e); the final matrix square is an ordinary matrix
product. It is not an elementwise square or a squared Frobenius norm.

`OverlapEvaluator` builds actual signed spin blocks from the stored channels.
It preserves (down,up) leg assignments, rather than treating all opposite-spin
channels as having the same spatial ordering. Repeated same-spin modes are
merged using kappa_new=kappa_old+d+kappa_old*d. Distinct modes in one basis
are orthogonal, so no cross term survives between them. Zero increments
can be skipped without deleting the channel itself.

Disjoint-spin blocks, one-body/one-body pairs in the joint h basis, and
same-Cholesky-basis pairs commute structurally. Their zero contribution is
skipped in the numerator, while their coefficient pair remains in `weight`.
Other pairs use lazily cached Q.T@X_gamma or X_gamma.T@X_eta overlaps, with
only small selected submatrices passed to the norm kernel. No dense channel
matrices or commutators are constructed by the overlap evaluator.

All bases must have orthonormal columns. Tiny negative squared norms relative
to the computed terms are clamped to zero; materially negative values raise
FloatingPointError. This is a roundoff safeguard, not a replacement for valid
overlaps. Near commuting pairs, use absolute as well as relative comparisons
against the dense reference.

The cache is bound to a fixed decomposition and owns snapshots of the bases.
Create a new evaluator when the gauge, amplitudes, or channel list changes.
The reference evaluator still visits O(N_channels**2) unordered pairs. Its
lazy basis-overlap cache can grow to O(r**2*n**2) storage; it does not promise
cheap full objectives merely because individual blocks have low rank.
There is no autodifferentiation or stochastic pair sampling. Step 4 below
adds partition optimization using the same overlap kernel.

### Run the step-3 comparisons

```sh
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
python -m ipie.hamiltonians.optimize_cholesky.tests.test_objective
```

The direct module command prints dense, overlap, and sor_base losses for the
three policies, then runs the new tests. The tests AST-load the **actual**
compute_total_commutativity function from the checked-out sor_base.py,
unchanged, and capture its printed diagnostics. A read-only view supplies
exactly the same channel matrices and coefficients. This avoids unrelated
SOR construction/import dependencies while testing the requested reference
function itself. It is not a replacement copy of that function.

Additional tests cover every channel pair, both partition maps, rotated and
identity gauges, signed low-rank coefficients, both spin contributions,
opposite-spin leg ordering, repeated modes, zero sources, one-channel edge
cases, raw/normalized weights, coefficient rescaling, invalid overlaps, and
negative-roundoff handling. The overlap path is also tested with dense channel
construction disabled. The stale step-1 test import has been updated from
the removed gauge module to your merged prepare module.

## Step 4: fixed-gauge partition optimization

`partitions.py` optimizes the unweighted squared `L_raw`, at fixed `tau1`,
`tau2`, Cholesky gauge/eigenpairs and one-body channels. The source ordering
and `t` coordinates are exactly those from step 2. One two-branch source
shares its coordinate across both signs; one spin quartet shares it across
all four spin products. Channel counts, positive unnormalized `a_i` values
and the leg product `d_left*d_right = -W_i*tau2` are preserved.

```python
from ipie.hamiltonians.optimize_cholesky import (
    partition_coefficients, optimize_partitions, raw_loss,
)

# decomposition is the existing output of build_fixed_decomposition(...).
coefficients = partition_coefficients(decomposition)
if coefficients:
    c = coefficients[0]
    print(c.A, c.B, c.C, c.gradient)
    print(c.value(c.t))  # current complete L_raw, not a single-pair loss

result = optimize_partitions(
    decomposition,
    max_sweeps=200,
    gradient_atol=1e-10,
    gradient_rtol=1e-8,
)
optimized = result.decomposition
print(result.status, result.sweeps, result.loss)
print(result.projected_gradient_norm, result.gradient_threshold)
assert abs(raw_loss(optimized) - result.loss) < 1e-12
```

The returned frozen `PartitionOptimizationResult` contains the updated
`FixedDecomposition`, final coefficients/gradient, bounds, convergence status
and a `PartitionSweep` history. Sweep 0 describes the input. Output arrays
are owned read-only snapshots; the input is unchanged. The output can be
warm-started with another `optimize_partitions(result.decomposition, ...)`.
This does not add a `sor_base` adapter or an outer gauge optimizer.

### Coordinate coefficients

For source i, hold every other coordinate fixed. Each affected single-particle
rotation has the form

```
u_c(s) = I + exp(s)*P_c + exp(-s)*Q_c,     c in S_i,
M_cj = [P_c, u_j],  N_cj = [Q_c, u_j],   j not in S_i.
```

`P_c`, `Q_c` contain the signed t=0 leg amplitudes and their actual spin
embeddings. The two legs are distinct orthogonal spin-orbital modes in one
Cholesky basis, so no quadratic single-particle increment survives. Internal
source commutators vanish. With Frobenius products summed over both spins,

```
A_i = sum_{c in S_i, j not in S_i} ||M_cj||_F**2 >= 0
B_i = sum_{c in S_i, j not in S_i} ||N_cj||_F**2 >= 0
C_i = L_unaffected + 2*sum <M_cj,N_cj>
L_raw(s) = C_i + A_i*exp(2*s) + B_i*exp(-2*s)
g_i = 2*A_i*exp(2*t_i) - 2*B_i*exp(-2*t_i).
```

Each affected/external unordered pair occurs once; there is no extra factor
1/2. `C_i` includes all unaffected pairs and can be negative. These are global
source coefficients, not coefficients obtained by optimizing individual
commutator pairs. The channel coefficients `a_i` do not multiply `L_raw`.

`PartitionCoefficients` stores `left_term=A_i*exp(2*t_i)` and
`right_term=B_i*exp(-2*t_i)` at its reference `t`, with properties `A`, `B`,
and `gradient`, plus `value(s)` and `derivative(s)`. Centered evaluation and
updates avoid forming extreme unscaled coefficients. Accessing `A` or `B`
raises `FloatingPointError` if it overflows; optimization can still use the
centered terms. `C_i` is evaluated as the full current loss minus those two
terms, with the usual cancellation limits of float64 arithmetic.

`partition_coefficients(decomposition)` returns the tuple for every source;
`partition_coefficients(decomposition, source_index)` returns one. Both
validate the source map once and use cached basis overlaps. The inner solver
reuses its workspace/cache and updates only the affected small spin blocks.
It performs no ERI reconstruction, eigendecomposition or dense rotation
construction. It still sums external channels for each source, and refreshes
the full pair sum once per sweep: it is a deterministic reference solver,
not a linear-scaling implementation.

### Feasible closed bounds

The step-2 decoder requires `abs(d) < amplitude_bound <= 1`. A finite
coordinate minimum need not exist on that open domain. The inner solver
instead uses a closed limit `rho=amplitude_limit < amplitude_bound`, default
`rho=(1-1e-8)*amplitude_bound`, and finite `max_abs_t=30`. For each source,

```
lower_i = max(-max_abs_t, log(abs(d_right(0))/rho))
upper_i = min( max_abs_t, log(rho/abs(d_left(0))))
```

A zero left/right leg contributes no corresponding amplitude bound. These
formulas cover both `balanced_log` and `anchored_log`; the former has symmetric
amplitude bounds for nonzero W. `partition_bounds(...)` returns the actual
read-only `(n_sources,2)` bounds used by the solver. The input t must already
lie in them, otherwise the solver raises a descriptive error. Infeasible
scales/bounds are errors; amplitudes are never individually clipped.

`max_abs_t` is an explicit numerical restriction, especially relevant for
`anchored_log` at W=0. A surviving leg may lower the loss by tending to zero
only as t tends to positive/negative infinity. The finite box makes this
case well-defined and keeps the existing anchored family intact. Convergence
at that wall certifies the stored bounded problem. Increasing `max_abs_t`
can lower its solution further. `balanced_log` has both legs zero at W=0,
so that source is flat and keeps its current t.

### Sequential updates and inner stopping

`analytic_partition_update(c, bounds_i)` gives the exact coordinate minimum:

```
A_i,B_i > 0 : clip(log(B_i/A_i)/4, lower_i, upper_i)
A_i=0, B_i>0: upper_i
A_i>0, B_i=0: lower_i
A_i=B_i=0  : retain the current feasible coordinate.
```

The implementation uses logarithms of the centered terms. During each sweep,
sources are visited in their fixed step-2 order, coefficients are recomputed
after every previous accepted update, and the branches change together.
Simultaneously assigning all old coordinate minima is a different procedure
and need not decrease the loss. A roundoff guard rejects tiny predicted
increases and raises on a material increase; a complete overlap-loss refresh
checks sweep descent and prevents drift from repeated subtractions.

After each sweep the solver recomputes every gradient at the final state of
that sweep. For `g=gradient_t`, the projected KKT residual is

```
r_i = g_i           in the interior
r_i = min(g_i,0)    at the lower bound
r_i = max(g_i,0)    at the upper bound
r_i = 0            if lower_i == upper_i.
```

It stops when

```
max(abs(r)) <= gradient_atol + gradient_rtol*initial_max_abs_r.
```

An active bound can therefore converge with a nonzero unconstrained gradient.
This criterion is in t coordinates; it is not a gradient in rescaled bounded
coordinates. A small loss decrease alone never triggers convergence.
`max_sweeps=0` evaluates the input without updates. Exhausting the limit
returns `converged=False, status="max_sweeps"` with the current residual.
Coordinate stationarity does not certify a global minimum of the general
joint partition problem. Weakly conditioned problems can require more than
the default 100 sweeps; inspect the status before an outer gauge update.

### Step-4 verification

```sh
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_partitions -v
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
```

Tests compare `A_i,B_i,C_i` with an independent dense isolated-leg commutator
oracle, profile values with the complete dense loss, gradients with finite
differences, and analytic minima with a scalar numerical minimizer. They
also verify sequential rather than simultaneous updates, descent, both
partition maps, all three policies, fresh stopping gradients, active bounds,
zero couplings, unchanged input/weights/gauge, full Hamiltonian reconstruction
and agreement with the actual `compute_total_commutativity` function after
optimization. The inner path is exercised with dense rotation construction,
eigendecomposition and ERI validation disabled.

## Step 5: analytic gauge gradients

The new modules are `derivatives.py` (small-kernel, eigensystem and gauge-map
pullbacks) and `gradients.py` (complete fixed-partition and relaxed gradients).
They use NumPy/SciPy float64 arithmetic, with no autodiff dependency. The
objective remains the unweighted squared `L_raw` from step 3.

### Derivative backend option

Select the derivative backend when constructing the engine:

```python
evaluator = GaugeEvaluator(
    problem, tau1=dt, tau2=dt,
    partition="balanced_log", backend="numpy",
)
print(evaluator.backend)  # "numpy", read-only
```

`backend` defaults to `"numpy"`, preserving existing calls and numerical
results. The accepted names are case-sensitive:

| Backend request | Current behavior |
| --- | --- |
| `"numpy"` | Existing analytic NumPy/SciPy derivatives |
| `"pytorch"` | Reserved; immediately raises `NotImplementedError` |
| Another string | Raises `ValueError` |
| A non-string | Raises `TypeError` |

The engine validates this choice before one-body setup, eigendecomposition
or inner optimization. The reserved path never imports PyTorch and never
silently falls back to NumPy. No PyTorch installation is needed.
The selection remains fixed for that engine instance; construct a new engine
to request a different backend.

The same optional keyword is available on standalone derivative entry points:

```python
fixed = fixed_partition_gradient(problem, decomposition, backend="numpy")
normalized = normalized_partition_gradient(
    problem, decomposition, backend="numpy",
)
relaxed = relaxed_gauge_gradient(problem, partition_result, backend="numpy")

gradient_theta = exponential_gauge_vjp(
    theta, r, fixed.gradient_O, reference=None, backend=evaluator.backend,
)
gradient_angles = givens_gauge_vjp(
    r, rotations, fixed.gradient_O, reference=None, backend=evaluator.backend,
)
```

Existing positional reference arguments for the two coordinate pullbacks
remain valid. The small overlap/eigh primitives remain explicit NumPy
helpers; backend resolution is kept outside the channel-pair loops.
`check_gauge_direction` uses the backend already selected by its evaluator
callable. Input preparation, channel construction, partition optimization
and value-only loss functions retain their current NumPy interfaces.

This is a derivative-computation setting, separate from the `dense`/`overlap`
loss method and from the global `ipie.utils.backend` array backend.
`derivatives.py` centralizes name validation. The current forward calculations,
SciPy operations and immutable NumPy snapshots are not a PyTorch autograd
graph. A future PyTorch implementation must supply a tensor-based forward
and derivative path, including the relaxed moving-bound chain rule; changing
the accepted backend name alone will not provide that support. Public engine
and derivative calls can keep the interface introduced here.

Backend compatibility tests:

```sh
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_backends -v
```

These tests verify default/explicit NumPy equality in all three gauge modes,
standalone and coordinate-pullback compatibility, immediate unsupported-backend
errors, invalid-name/type errors, and operation in a fresh interpreter that
blocks every `torch` import.

### Evaluating a trial gauge

```python
import numpy as np
from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator, exponential_gauge, exponential_gauge_vjp,
)

# problem is the existing prepare_problem(...) / load_problem(...) result.
evaluator = GaugeEvaluator(
    problem, tau1=dt, tau2=dt, partition="balanced_log",
)
r = problem.data.nchol
theta = np.zeros(r*(r-1)//2)
O = exponential_gauge(theta, r)

# Hold every physical log ratio t_i fixed.
fixed = evaluator.fixed(O, t=0.0)
print(fixed.loss, fixed.partition_gradient)
gradient_theta = exponential_gauge_vjp(theta, r, fixed.gradient_O)
```

For the two-spatial-orbital `(1,1)` case documented separately, the balanced
`t=0` choice is already a global partition minimum at every admissible gauge.
Use `evaluator.fixed(O, t=0.0)` directly; no inner solve is required.

For general systems, evaluate the relaxed loss and gradient together:

```python
relaxed = evaluator.relaxed(
    O, t0=None, max_sweeps=500,
    gradient_atol=1e-11, gradient_rtol=1e-9,
    amplitude_limit=None, max_abs_t=30.0,
)
gradient_theta = exponential_gauge_vjp(theta, r, relaxed.gradient_O)
print(relaxed.loss, relaxed.inner_result.status,
      relaxed.inner_result.projected_gradient_norm)
optimized = relaxed.decomposition
```

Every call to `relaxed` performs an inner solve at that specific trial gauge.
An unfinished inner solve raises `InnerConvergenceError`, whose `.result`
contains the step-4 diagnostics. Increase the sweep limit or adjust the
specified tolerances explicitly. With `t0=None`, start at the current interval
midpoints. A supplied warm-start t vector is projected onto the current
intervals before construction; amplitudes are never clipped separately.

`GaugeEvaluator` caches the fixed one-body channels and source topology. Each
trial rotates the original factors and diagonalizes the resulting batch once.
It performs no full ERI reconstruction or dense-channel construction. The
fast trial `GaugeState` has `diagnostics=None`; use `validate_gauge_state`
for an independent full validation when desired, outside the loss loop.
The gradient still visits channel pairs and caches basis overlaps; this is
a deterministic reference implementation, not a linear-scaling optimizer.

Standalone functions also accept existing step-2/4 objects:

```python
fixed = fixed_partition_gradient(problem, decomposition)
relaxed = relaxed_gauge_gradient(problem, partition_result)
```

`relaxed_gauge_gradient` checks a freshly computed partition KKT residual
and verifies the saved bounds and loss. Step-4 results now record the actual
`amplitude_limit` and `max_abs_t`, so a nondefault inner bound is not silently
replaced by a default when taking its derivative.

### Fixed t, fixed normalized coordinates, and relaxed derivatives

There are three intentionally separate interfaces:

| Method | Quantities held fixed | Gauge derivative |
| --- | --- | --- |
| `evaluator.fixed(O, t=...)` | Physical t vector | Partial derivative of L(O,t) |
| `evaluator.fixed_z(O, z=..., ...)` | Normalized z vector | Includes the interval map |
| `evaluator.relaxed(O, ...)` | Inner problem definition | Envelope derivative at the returned inner solution |

The step-4 intervals are `lower_i(O) <= t_i <= upper_i(O)`. To use a fixed
domain for both balanced and asymmetric anchored partitions, define

```
c_i(O) = (lower_i+upper_i)/2
h_i(O) = (upper_i-lower_i)/2
t_i = c_i(O)+h_i(O)*z_i,     -1 <= z_i <= 1.
```

The fixed-z chain rule is

```
derivative_O L(O,t(O,z))
    = partial_O L(O,t) + sum_i g_i*(derivative_O c_i+z_i*derivative_O h_i)
g_i = partial L / partial t_i.
```

Equivalently, the endpoint adjoints are `g_i*(1-z_i)/2` for the lower bound
and `g_i*(1+z_i)/2` for the upper bound. These derivatives include the actual
amplitude limits, asymmetric anchored intervals and clipping by the finite
`max_abs_t` cap. A cap-controlled endpoint is constant and has zero derivative.
With uncapped balanced intervals `[-T_i,T_i]`, this reduces to the previously
documented correction `sum_i g_i*z_i*derivative_O T_i`.

At a smooth converged inner minimum, the envelope theorem uses this fixed-z
derivative. It does not require differentiating through the coordinate-sweep
algorithm. For an interior optimum `g_i=0`; at a lower active wall `g_i>=0`
and the correction is `g_i*derivative_O lower_i`; at an upper active wall
`g_i<=0` and it is `g_i*derivative_O upper_i`. Therefore using a fixed-t
gradient for a relaxed loss generally fails at moving active bounds.

For finite inner tolerances the returned derivative is an approximate
envelope gradient. The implementation retains the small interior residual
terms in the fixed-z chain rule. Gradient checks should tighten the inner
tolerance and re-solve the SAME local branch at every perturbed gauge.
KKT convergence does not certify a global inner minimum in the general case.
The relaxed gradient assumes a regular consistently tracked local minimum;
changes of solution branch can make the value function nonsmooth.

### Analytic reverse pass

For an overlap block `C=V.T@Z`, the norm derivative is propagated into both
bases: `bar_V=Z@bar_C.T`, `bar_Z=V@bar_C`. Actual signed channel amplitudes
are differentiated before their shared-source pullback. Zero legs are kept
in the differentiable blocks: a zero amplitude can have a nonzero derivative
when the same spin block contains another active mode.

`commutator_norm2_derivatives(d,e,C)` returns the norm and the three polynomial
derivatives with respect to d, e and C. It differentiates the analytic norm
formula, not the negative-roundoff clamp. Same-basis channels commute for
every gauge and are skipped structurally.

For `L=X@diag(ell)@X.T`, `eigh_vjp` implements

```
M = X.T@bar_X
D[a,b] = 1/(ell[b]-ell[a]) for a != b; D[a,a] = 0
bar_L = X @ (diag(bar_ell)+sym(D*M)) @ X.T
```

The rotated-factor derivative then gives the ambient gauge derivative
`E[gamma,eta] = sum_pr bar_L[gamma,p,r]*L_original[eta,p,r]`.
For both fixed and moving-bound objectives, `GaugeGradient` returns:

| Field | Meaning |
| --- | --- |
| `loss`, `decomposition` | Value and the exact corresponding channel snapshot |
| `gradient_O` | Ambient gauge derivative E |
| `fixed_gradient_O` | Contribution holding physical t fixed |
| `bound_correction_O` | Moving-interval contribution; sum with the previous field gives E |
| `tangent_gradient` | Frobenius-metric tangent projection at O |
| `generator_gradient` | `skew(E@O.T)`; tangent gradient is this matrix times O |
| `partition_gradient` | Physical t gradient, not a z gradient |
| `eigenvalue_gradient`, `eigenvector_gradient` | Spectral adjoints; the former includes bound response |
| `rotated_cholesky_gradient` | Symmetric factor adjoints including bound response |
| `normalized_partitions`, `inner_result` | z for fixed-z/relaxed modes; inner diagnostics for relaxed mode |

All derivative arrays are owned read-only snapshots. For a left tangent
direction `delta_O=K@O`, with `K.T=-K`, the directional derivative is
`sum(gradient_O*(K@O))`. The local plane-angle derivative is **twice** the
upper entry of `generator_gradient`, because the skew generator has both
an upper and a lower varying entry. This factor is distinct from any
choice of manifold metric.

### Exponential and ordered Givens coordinates

`exponential_gauge_vjp(theta,r,E,reference=None)` uses the exact Frechet
adjoint of `expm(S(theta))@reference`. A local tangent generator is not
the gradient in accumulated global exponential coordinates at nonzero theta.

`givens_gauge_vjp(r,rotations,E,reference=None)` differentiates the exact
left-multiplication order in `givens_gauge`, including repeated plane indices.
The returned vector has one entry per rotation in the original list order.
Reflection references and the zero-parameter r=1 exponential case are supported.

### Smooth-branch requirements

`GradientOptions` configures per-factor eigengap and near-zero tolerances.
The implementation raises `GaugeRegularityError` for repeated/near-degenerate
Cholesky eigenvalues; it never adds an epsilon to their differences. It also
rejects zero/near-zero eigenvalues used by `balanced_log`, where the square
root/absolute-value sign branch need not have a smooth gauge derivative.
`anchored_log` supports a simple zero eigenvalue because its amplitudes are
linear in the individual eigenvalues; the other eigengaps must still be regular.

For moving-bound derivatives, an amplitude endpoint tied with the numerical
cap or a collapsed interval is reported explicitly when needed by the chain
rule. Active-set or eigenvalue branch changes can require directional rather
than ordinary smooth derivatives. These cases are not silently smoothed.

### Checking the two gradient definitions separately

```python
from ipie.hamiltonians.optimize_cholesky import check_gauge_direction

# K is any skew-symmetric r*r matrix. Keep the same t on every trial.
fixed_check = check_gauge_direction(
    lambda matrix: evaluator.fixed(matrix, t=0.0), O, K,
)

# Re-relax at EVERY positive and negative finite-difference point.
seed = relaxed.decomposition.two_body.t
relaxed_check = check_gauge_direction(
    lambda matrix: evaluator.relaxed(
        matrix, t0=seed, max_sweeps=500,
        gradient_atol=1e-12, gradient_rtol=0,
    ), O, K,
)
for row in relaxed_check.differences:
    print(row.step, relaxed_check.analytic,
          row.finite_difference, row.absolute_error)
```

Use the same bound settings at every point if the center used nondefault
`amplitude_limit` or `max_abs_t`. The helper uses the orthogonality-preserving
path `expm(step*K)@O` and reports a sequence of central differences, rather
than a single step-size result.

```sh
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_gradients -v
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
```

The step-5 tests separately cover fixed t versus independent dense differences,
fixed z with moving and capped intervals, and re-relaxed envelope gradients.
They include all three sectors, both partition maps, active walls with nonzero
raw t gradients, interior solutions, exact coordinate-map pullbacks, simple
zero anchored legs, regularity failures and Hamiltonian reconstruction.
At active amplitude bounds the tests also verify that omitting the correction
produces an appreciably wrong derivative.

Step 4 received one numerical fix needed for tight relaxed-gradient checks:
`PartitionCoefficients.change(t)` uses `expm1` for tiny coordinate changes.
Subtracting four nearly equal profile terms could previously hide a legitimate
small decrease and stall the inner solve above its requested gradient tolerance.

## Step 6: outer optimization in fixed exponential coordinates

`outer.py` supplies the fixed-exponential reduced objective. The concrete
`LBFGSOptimizer` in `solvers/lbfgs.py` minimizes it with Armijo backtracking,
using the shared `OuterOptimizer` lifecycle in `solvers/base.py`. The
reference matrix stays fixed for the entire L-BFGS run:

```
O(theta) = expm(S(theta)) @ reference
F(theta) = L_raw(O(theta), t_star(O(theta)))
```

`S` uses the same strict-upper-triangle packing as `exponential_gauge`.
Every new trial gauge independently relaxes the complete partition block
and computes the step-5 moving-bound envelope gradient. This includes
rejected line-search points. No finite-difference outer gradient or
differentiation through the partition-sweep algorithm is used.

### Run a nested outer solve

```python
from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator, InnerOptions, ExponentialGaugeObjective,
    LBFGSOptions, LBFGSOptimizer,
)

engine = GaugeEvaluator(
    problem, tau1=dt, tau2=dt, partition="balanced_log", backend="numpy",
)

def report(row):
    print(
        f"outer {row.iteration:3d}: L_raw={row.loss:.10e} "
        f"g_theta={row.gradient_norm:.3e} "
        f"g_tangent={row.tangent_gradient_norm:.3e} "
        f"inner_sweeps={row.inner_sweeps}"
    )

objective = ExponentialGaugeObjective(
    engine,
    reference=None,    # identity by default
    inner_t0=None,     # each trial starts at its own interval midpoints
    inner_options=InnerOptions(
        max_sweeps=500, gradient_atol=1e-12, gradient_rtol=0,
        amplitude_limit=None, max_abs_t=30.0,
    ),
)
solver = LBFGSOptimizer(
    objective,
    options=LBFGSOptions(
        max_iterations=100, max_evaluations=1000,
        gradient_atol=1e-8, gradient_rtol=1e-6,
        memory=10, max_step_norm=1.0,
    ),
)
result = solver.optimize(initial=None, callback=report)  # zero theta by default

print(result.status, result.message)
print(result.initial_loss, result.loss, result.gradient_norm)
print(result.partition_result.projected_gradient_norm)
optimized = result.decomposition
O_optimized = result.O
```

`result.loss` is the raw squared commutator loss at the final accepted gauge
and its final relaxed partitions. `result.initial_loss` is already the
**relaxed** loss at the initial gauge; it is not an unrelaxed baseline.
The returned `decomposition` contains the ordinary source/channel objects,
positive weights and amplitudes needed by the future SOR adapter.

`OptimizerOptions` contains common stopping, budget and final-validation
settings. `LBFGSOptions` extends it with memory and line-search settings.
`InnerOptions` configures partition relaxation; `GradientOptions` is still
supplied to the engine for derivative regularity guards.

Existing calls to `optimize_gauge(engine, options=OuterOptions(...), ...)`
continue to work and delegate to `LBFGSOptimizer`. That convenience function
also accepts `LBFGSOptions` or common `OptimizerOptions` (the latter uses
default L-BFGS-specific settings). The legacy `OuterOptions` constructor,
including its positional field order, and the result fields are preserved.
Choose other solvers by constructing their classes; the parent does not
dispatch on a solver string.

### Solver class boundaries

| File/class | Responsibility |
| --- | --- |
| `outer.py`: `ExponentialGaugeObjective` | Fixed reference and coordinate map, nested partitions, exact pullback, joint cache, final gauge validation |
| `solvers/base.py`: `OuterOptimizer` | Fresh run state, per-run counters, budgets, accepted-state bookkeeping, callbacks, stopping, final validation |
| `solvers/lbfgs.py`: `LBFGSOptimizer` | Packed-theta proposals, two-loop recursion, Armijo acceptance, curvature memory, exponential-coordinate result |
| `solvers/gdm.py`: `GDMOptimizer` | Reserved subclass; currently raises `NotImplementedError` before evaluating the objective |

The parent neither assumes exponential coordinates nor computes directions,
chooses trial paths, performs line searches, or interprets curvature history.
It takes an objective and exposes three abstract hooks:

```python
class NewOptimizer(OuterOptimizer):
    def _initialize(self, initial):
        # Return (initial_evaluation, fresh_solver_state).
        ...

    def _step(self, point, run, iteration):
        # Return OptimizerStep(evaluation=accepted_point, ...), or
        # OptimizerStep(status="reason", message="...", ...) on failure.
        ...

    def _build_result(self, point, summary):
        # Package this solver's point plus common summary metadata.
        ...
```

`run.solver_state` is opaque to the parent. The child must respect the
remaining budget during any multi-evaluation step via `_budget_exhausted(run)`.
Only an accepted `OptimizerStep.evaluation` replaces the stored point.
The parent passes common `OptimizerIteration` records to callbacks; a child
can extend `_iteration_record`, as L-BFGS does for memory diagnostics.

The objective provides `evaluate(parameters)` and a cumulative `evaluations`
counter. Its points provide `loss`, `gradient_norm`, `tangent_gradient_norm`,
and `gauge_gradient.inner_result` for partition diagnostics. No `theta` or
`reference` is required by the parent. The optional objective counters are
`failed_evaluations`, `cache_hits` and `inner_sweeps`. With final validation
enabled, the objective also provides `validate(point)` and the child's result
is a dataclass with a `validation_report` field.

`GDMOptimizer` is an extension point, not an available numerical solver yet.
Its future implementation can use matrix-valued gauge points, its own
manifold objective, geometric paths/retractions and transported directions
or history. Those choices will live in the GDM child and its objective,
without adding GDM branches to the parent. No claim of a Q-Chem-equivalent
GDM algorithm is made by this placeholder.

The callback receives immutable scalar `OuterIteration` records at iteration
zero and after each accepted step, never at rejected trials. Raising
`StopIteration` stops at the current accepted state with
`status="callback_stopped"`. Other callback exceptions propagate.

For your two-orbital `(1,1)` balanced setting, interval midpoints are zero
and the inner gradient already vanishes. The generic nested call therefore
reports zero inner sweeps without requiring a separate outer algorithm.
The gauge objective may still vary, or it may already be stationary at the
initial gauge. Single-factor problems have no continuous gauge parameters
and are evaluated once.

### Reference, seed and cache semantics

The `reference` argument is the fixed right factor in `exp(S(theta))@reference`.
To start from an existing orthogonal `O_initial`, use `reference=O_initial`
with the default zero theta. Alternatively, pass an existing packed theta
with identity reference. Passing nonzero theta and a nonidentity reference
applies both transformations in the stated order.

L-BFGS never recenters this reference or reuses curvature information
in a different coordinate chart. Reflection references are supported. A new
call to `optimize_gauge`, or to the same `solver.optimize(...)`, starts with
empty curvature history. Reusing a solver/objective preserves the objective's
one-entry cache. Each result and evaluation budget uses counters relative to
the start of that optimize call, so a cached initial point can report zero
fresh evaluations and one cache hit.

`inner_t0` is copied once and stays fixed across the entire run. It can be a
scalar or one log ratio per source. Every trial projects that same seed into
its own feasible intervals. With `None`, every trial independently uses its
current interval midpoints. Neither accepted nor rejected trial partitions
replace the seed. This avoids a history-dependent objective caused by
feeding different initial guesses to identical parameter values. General
inner solves can still select different local branches as the gauge changes;
the envelope assumptions from step 5 remain necessary.

`ExponentialGaugeObjective` can also be used directly:

```python
from ipie.hamiltonians.optimize_cholesky import ExponentialGaugeObjective

objective = ExponentialGaugeObjective(
    engine, reference=None, inner_options=InnerOptions(), inner_t0=None,
)
point = objective.evaluate(theta)
value, gradient_theta = objective.value_and_grad(theta)
```

The one-entry cache stores a matching `ExponentialEvaluation`: copied theta,
value, exact coordinate gradient, gauge derivative and converged inner
snapshot. A repeated theta reuses that joint record. `value_and_grad` returns
a writable gradient copy for external optimizers. The reference, inner
configuration and seed are read-only. Do not modify the underlying engine
configuration during a run; create a new objective instead.

### L-BFGS and line-search behavior

The exact coordinate gradient is obtained with
`exponential_gauge_vjp(theta,r,E,reference,backend=engine.backend)`, where E
already includes partition-bound response. Search directions and curvature
pairs live in the global packed-theta space:

```
s_k = theta_(k+1) - theta_k
y_k = grad_theta_(k+1) - grad_theta_k
```

There is no need to reset history just because the partitions changed:
both endpoint gradients describe the same nested objective definition.
The standard two-loop L-BFGS recursion stores at most `memory` pairs. A
pair is retained only when `s.T@y` is positive and exceeds
`curvature_rtol*norm(s)*norm(y)`. A nonfinite or non-descent direction clears
the history and falls back to steepest descent.

Without curvature history, steepest descent is normalized to unit coordinate
length so a small numerical scale of `L_raw` does not force tiny initial
steps. Trial displacement norms are capped by `max_step_norm`. The line search
starts at `initial_step`, shrinks by `backtrack`, and accepts only

```
F(theta + alpha*p) <= F(theta) + armijo*alpha*(grad_theta @ p).
```

This reference implementation enforces Armijo decrease and filters curvature
explicitly; it does not require a strong-Wolfe line search.

An infeasible trial interval raises `GaugeFeasibilityError`, which remains a
`ValueError` subclass for compatibility. That error, `GaugeRegularityError`,
`InnerConvergenceError`, `FloatingPointError` and `numpy.linalg.LinAlgError`
reject the trial and trigger backtracking. No fake zero gradient or infinite
loss is passed off as a valid evaluation. Unimplemented backends and
programming/configuration errors propagate. Failure at the initial point
raises immediately, since no valid accepted result exists yet.

### Convergence, status and diagnostics

At each accepted state the optimizer checks both

```
norm(gradient_theta, infinity) <= gradient_atol
                               + gradient_rtol*initial_coordinate_norm
norm(tangent_gradient, Frobenius) <= gradient_atol
                                  + gradient_rtol*initial_tangent_norm.
```

The physical tangent check matters because accumulated exponential
coordinates can become singular/redundant. A small coordinate gradient alone
is reported as `coordinate_stalled` if the physical tangent criterion fails.
Small loss changes or small steps alone never imply convergence.

| Status | Meaning |
| --- | --- |
| `converged` | Both outer gradient tests passed; the stored inner solve also converged |
| `max_iterations` | Accepted-iteration limit reached |
| `max_evaluations` | Fresh trial-evaluation budget exhausted |
| `line_search_failed` | No admissible Armijo-decreasing trial was accepted |
| `coordinate_stalled` | Coordinate test passed but physical tangent test failed |
| `stalled` | A usable numerical descent direction could not be represented |
| `callback_stopped` | Callback requested stopping at an accepted state |

All nonconverged exits preserve the last **accepted** point. A rejected trial
cannot overwrite its theta, loss, gradient or channel decomposition.
`history` contains accepted-state diagnostics; `trials` contains scalar
accepted/rejected trial records and rejection reasons. Its `attempt` field
numbers line-search trials starting at one within each iteration. Inspect these if a
line search stops:

```python
for row in result.trials[-5:]:
    print(row.iteration, row.step_size, row.accepted, row.reason)
```

`evaluations` counts fresh oracle calls within this run, including recoverable
failed calls and an uncached initial evaluation; cache hits are recorded
separately. `inner_sweeps`
sums sweeps reported by returned inner snapshots, including snapshots attached
to post-inner gradient failures. An exceptional inner run that produces no
snapshot cannot report its partial work. Full gauge/ERI validation is run
once on the final accepted state by default and returned as
`result.validation_report`. Set `LBFGSOptions(validate_final=False)` to omit
that explicit final reference check; it is never repeated inside the loops.

This is a local smooth-branch optimizer. Nonsmooth eigenvalue-zero minima or
active-set/inner-branch transitions can cause honest nonconverged exits even
after substantial descent. It does not certify a global gauge minimum.
Use sufficiently tight inner tolerances for the requested outer accuracy.

The outer parameterization is unconstrained; feasibility is handled by
rejecting inadmissible trials. This is not a separate constrained-gauge KKT
solver at the boundary of the admissible gauge domain. A sufficiently small
tau2, for example the conservative all-gauge bound documented in the note,
avoids that domain restriction:

```
tau2*lambda_max(G) < rho**2,
G[gamma,eta] = trace(L_original[gamma] @ L_original[eta]).
```

Here rho is the actual closed amplitude limit: the explicit `amplitude_limit`,
or `(1-1e-8)*engine.amplitude_bound` when that option is None.

### Step-6 verification

```sh
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_outer -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_outer_controlled -v
python -m unittest ipie.hamiltonians.optimize_cholesky.tests.test_solver_classes -v
python -m unittest discover -s ipie/hamiltonians/optimize_cholesky/tests -v
```

Controlled analytic quadratic oracles test optimizer convergence, Armijo
descent, curvature filtering, fallback directions, callback stopping,
iteration/evaluation budgets, failure recovery and accepted-state retention.
These tests make no Hamiltonian-validation claim. Separate physical tests
verify nested descent, dense-loss/gradient parity, moving-bound response,
final ERI/Hamiltonian reconstruction in all three sectors and both maps,
the two-orbital zero-inner-sweep shortcut, a nontrivial physical converged
case, and a singular exponential chart with a nonzero physical gradient.

The class tests compare the direct L-BFGS API with the compatibility wrapper
on a physical Hamiltonian, run the generic parent with a test-only matrix
objective that has no exponential attributes, verify final-validation calls,
check repeated-run memory/budget/counter behavior, and assert that GDM fails
explicitly before doing any objective work.







