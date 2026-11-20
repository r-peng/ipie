"""Analytic gauge gradients and explicit derivative-backend selection."""

import numpy as np
import math
from scipy.linalg import expm_frechet, expm
from .models import (
    GaugeRegularityError,
    GradientOptions,
    FixedDecomposition,
    GaugeDifference,
    GaugeGradient,
    GaugeGradientCheck,
    GaugeState,
    InnerConvergenceError,
    PartitionOptimizationResult,
    PreparedProblem,
)
from .loss import ModeSpace, loss_kind_name, loss_pullback
from .partitions import (
    _readonly,
    _partition_intervals,
    interval_logs,
    _partition_limits,
    _inner_options,
    _solve_partitions,
    _validate_partition_state,
    projected_partition_gradient,
)
from .problem import (
    givens_gauge,
    skew_from_parameters,
    _dimension,
    _options,
    _real_array,
    _gauge_array,
)
from dataclasses import replace
from .channels import (
    _two_body_from_amplitudes,
    build_one_body_channels,
    enumerate_sources,
    fixed_partition_amplitudes,
    one_body_at_shift,
    positive_scale,
    shift_pair,
    source_eigenvalues,
)
from .shift import check_one_body_shift, shift_bounds, shift_coefficients


# Derivative backend, elementary pullbacks, and coordinate maps

def _resolve_derivative_backend(backend):
    """Validate a backend request before entering numerical work.

    Keep this outside the channel-pair kernels. A future PyTorch backend must
    implement the differentiable forward/gradient path at the gauge-gradient
    entry points; accepting the name alone would not create an autograd graph.
    """
    if not isinstance(backend, str):
        raise TypeError("Derivative backend must be a string: 'numpy' or 'pytorch'")
    if backend == "numpy":
        return backend
    if backend == "pytorch":
        raise NotImplementedError(
            "Derivative backend 'pytorch' is not implemented yet; use backend='numpy'. "
            "The current implementation uses analytic NumPy/SciPy derivatives."
        )
    raise ValueError(
        f"Unknown derivative backend {backend!r}; expected 'numpy' or 'pytorch' "
        "('pytorch' is reserved and not implemented yet)"
    )


def _gradient_options(options):
    if options is None:
        return GradientOptions()
    if not isinstance(options, GradientOptions):
        raise TypeError("options must be GradientOptions")
    return options


def eigh_vjp(eigenvalues, eigenvectors, grad_values, grad_vectors, *, options=None):
    """Symmetric-matrix pullback for one simple real eigensystem.

    If M=X.T@bar_X, D_ab=(lambda_b-lambda_a)^(-1) for a!=b,
    bar_L=X[diag(bar_lambda)+sym(D*M)]X.T. Near degenerate eigenvalues are
    rejected, never separated with an epsilon. The diagonal of D is zero.
    """
    options = _gradient_options(options)
    ell, X, ge, gX = (_real_array(x,n) for x,n in (
        (eigenvalues,"eigenvalues"),(eigenvectors,"eigenvectors"),
        (grad_values,"grad_values"),(grad_vectors,"grad_vectors")))
    if ell.ndim != 1 or ge.shape != ell.shape or X.shape != (ell.size,ell.size) or gX.shape != X.shape:
        raise ValueError("Expected eigenvalue vectors and square eigenvector/adjoint matrices")
    if not np.allclose(X.T@X,np.eye(ell.size),atol=1e-12,rtol=1e-10):
        raise ValueError("Eigenvectors must be orthonormal")
    gaps = _eigengaps(ell[None])
    try:
        _check_eigengaps(ell[None],options,gaps)
    except GaugeRegularityError:
        raise GaugeRegularityError("Repeated or near-degenerate Cholesky eigenvalues; "
                                   "the simple-eigenpair gradient is not defined reliably") from None
    return _readonly(_eigh_pullback(ell[None],X[None],ge[None],gX[None],gaps)[0])


def exponential_gauge_vjp(theta, nchol, gradient_O, reference=None):
    """Gradient in the packed global exponential coordinates of prepare.py.

    O=expm(S(theta))@reference. Uses the exact Frechet adjoint; the local
    skew generator gradient is not the global theta gradient away from zero.
    O itself is not recomputed: the caller already formed it once for this
    gauge, so only theta, reference and gradient_O are checked here.
    """
    r = _dimension(nchol)
    ref = np.eye(r) if reference is None else _gauge_array(reference,r,_options(None))
    S = skew_from_parameters(theta,r)
    E = _real_array(gradient_O,"gradient_O")
    if E.shape != (r,r):
        raise ValueError("gradient_O must have shape (nchol,nchol)")
    adjoint = expm_frechet(S.T,E@ref.T,compute_expm=False)
    rows, cols = np.triu_indices(nchol,1)
    return _readonly(adjoint[rows,cols]-adjoint[cols,rows])


def givens_gauge_vjp(nchol, rotations, gradient_O, reference=None):
    """Angle gradient in the exact ordered Givens sequence of prepare.py."""
    rotations = tuple(rotations)
    O = givens_gauge(nchol,rotations,reference)
    E = _real_array(gradient_O,"gradient_O")
    if E.shape != O.shape:
        raise ValueError("gradient_O must have shape (nchol,nchol)")
    state = np.eye(nchol) if reference is None else np.array(reference,dtype=float,copy=True)
    prefixes = []
    for p,q,angle in rotations:
        prefixes.append((state[p].copy(),state[q].copy()))
        c,s = np.cos(angle),np.sin(angle)
        rp,rq = prefixes[-1]
        state[p],state[q] = c*rp+s*rq,-s*rp+c*rq
    adjoint = E.copy()
    gradient = np.zeros(len(rotations))
    for i in reversed(range(len(rotations))):
        p,q,angle = rotations[i]
        c,s = np.cos(angle),np.sin(angle)
        rp,rq = prefixes[i]
        ep,eq = adjoint[p].copy(),adjoint[q].copy()
        gradient[i] = np.sum(ep*(-s*rp+c*rq))+np.sum(eq*(-c*rp-s*rq))
        adjoint[p],adjoint[q] = c*ep-s*eq,s*ep+c*eq
    return _readonly(gradient)


# Gauge loss pullback, moving bounds, and evaluator engine

def _coordinates(value, count, name):
    values = _real_array(value,name)
    if values.ndim == 0:
        return np.full(count,float(values))
    if values.shape != (count,):
        raise ValueError(f"{name} must be a scalar or length-{count} vector")
    return values.copy()


def _loss_pullback(decomposition, loss_kind="weighted", check=True, space=None):
    """The loss_kind loss, channel-leg adjoints (T,2) and basis adjoints dL/dX (r,n,n).

    Mode-space evaluation (loss.loss_pullback); the leg adjoints are with
    respect to the actual channel amplitudes d.
    """
    return loss_pullback(decomposition, loss_kind, check, space)


def _validate_state(problem,decomposition):
    if not isinstance(problem,PreparedProblem):
        raise TypeError("problem must be a PreparedProblem")
    _validate_partition_state(decomposition)
    two = decomposition.two_body
    if two.sources != enumerate_sources(problem):
        raise ValueError("Decomposition sources do not match the prepared problem")
    state = two.gauge
    O = _gauge_array(state.O,problem.data.nchol,problem.report.options)
    expected = (O@problem.data.chol.reshape(problem.data.nchol,-1)).reshape(problem.data.chol.shape)
    atol,rtol = problem.report.options.atol,problem.report.options.rtol
    if np.shape(state.chol) != expected.shape or not np.allclose(state.chol,expected,atol=atol,rtol=rtol):
        raise ValueError("Gauge factors do not match O times the original Cholesky factors")
    ell,X = state.eigenvalues,state.X
    if np.shape(ell) != expected.shape[:2] or np.shape(X) != expected.shape:
        raise ValueError("Invalid Cholesky eigensystem dimensions")
    if not np.allclose((X*ell[:,None,:])@np.swapaxes(X,1,2),expected,atol=atol,rtol=rtol):
        raise ValueError("Stored Cholesky eigenpairs do not reconstruct the rotated factors")
    check_one_body_shift(decomposition.one_body)


def _check_zero_eigenvalues(eigenvalues,sources,partition,options):
    """balanced_log needs |ell| above zero_atol+zero_rtol*max|ell_g| on every source mode.

    Depends on the gauge eigenvalues only, so GaugeEvaluator runs it before any
    inner work. Anchored amplitudes are differentiable at a simple zero.
    """
    if partition != "balanced_log" or len(sources) == 0:
        return
    ell = np.asarray(eigenvalues,dtype=np.float64)
    tolerance = options.zero_atol+options.zero_rtol*np.max(np.abs(ell),axis=1,initial=0.)
    small = np.abs(ell) <= tolerance[:,None]
    bad = small[sources.bix,sources.modes[:,0]] | small[sources.bix,sources.modes[:,1]]
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise GaugeRegularityError(
            f"Source {i}, factor {int(sources.bix[i])}: balanced amplitudes meet a zero/near-zero "
            "eigenvalue; a smooth sign branch is required")


def _eigengaps(eigenvalues):
    """gap[g,a,b] = ell[g,b]-ell[g,a] and the off-diagonal mask."""
    ell = np.asarray(eigenvalues,dtype=np.float64)
    return ell[:,None,:]-ell[:,:,None],~np.eye(ell.shape[1],dtype=bool)


def _check_eigengaps(eigenvalues,options,gaps=None):
    """Every factor needs simple eigenvalues: |gap| > gap_atol+gap_rtol*max|ell_g|.

    The same test as eigh_vjp, for all factors at once; the first failing
    factor is reported.
    """
    ell = np.asarray(eigenvalues,dtype=np.float64)
    gap,mask = _eigengaps(ell) if gaps is None else gaps
    tolerance = options.gap_atol+options.gap_rtol*np.max(np.abs(ell),axis=1,initial=0.)
    bad = np.any((np.abs(gap) <= tolerance[:,None,None]) & mask,axis=(1,2))
    if np.any(bad):
        g = int(np.flatnonzero(bad)[0])
        raise GaugeRegularityError(f"Factor {g}: Repeated or near-degenerate Cholesky eigenvalues; "
                                   "the simple-eigenpair gradient is not defined reliably")


def check_gauge_regularity(gauge,sources,partition,options=None):
    """The smooth-derivative checks that depend on the gauge only, in the order _assemble applies them.

    Balanced zero eigenvalues first, then eigengaps. GaugeEvaluator calls this
    right after diagonalizing a trial gauge, so a singular trial is rejected
    before the inner solve instead of after it.
    """
    options = _gradient_options(options)
    _check_zero_eigenvalues(gauge.eigenvalues,sources,partition,options)
    _check_eigengaps(gauge.eigenvalues,options)


def _eigh_pullback(eigenvalues,eigenvectors,grad_values,grad_vectors,gaps=None):
    """eigh_vjp for all factors at once, without input checks (eigengaps already checked)."""
    X = eigenvectors
    Xt = np.swapaxes(X,1,2)
    gap,mask = _eigengaps(eigenvalues) if gaps is None else gaps
    inverse = np.zeros_like(gap)
    np.divide(1.,gap,out=inverse,where=mask)
    term = (Xt@grad_vectors)*inverse
    middle = (term+np.swapaxes(term,1,2))/2
    diagonal = np.arange(X.shape[1])
    middle[:,diagonal,diagonal] += grad_values
    result = X@middle@Xt
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite symmetric-eigh derivative")
    return (result+np.swapaxes(result,1,2))/2


def _interleaved(index_columns,weight_columns,active=None,size=0):
    """bincount of per-source contributions, accumulated in source order.

    Column k of row i is the k-th contribution of source i, as the per-source
    loop added them; inactive entries are skipped.
    """
    index = np.stack(index_columns,axis=1).reshape(-1)
    weight = np.stack(weight_columns,axis=1).reshape(-1)
    if active is not None:
        keep = np.stack(active,axis=1).reshape(-1)
        index,weight = index[keep],weight[keep]
    return np.bincount(index,weights=weight,minlength=size)


def _amplitude_pullback(two,g_amplitude,options,checked=False):
    """(dL/d ell, dL/dt) from the channel-leg adjoints, for every source at once.

    checked=True means the balanced zero-eigenvalue test already ran.
    """
    ell = np.asarray(two.gauge.eigenvalues,dtype=np.float64)
    r,n = ell.shape
    sources = two.sources
    count = len(sources)
    if not checked:
        _check_zero_eigenvalues(ell,sources,two.partition,options)
    signed = sources.channel_sign[:,None]*g_amplitude
    gleft = np.bincount(sources.channel_source,weights=signed[:,0],minlength=count)
    gright = np.bincount(sources.channel_source,weights=signed[:,1],minlength=count)
    left,right = two.amplitudes[:,0],two.amplitudes[:,1]
    gt = gleft*left-gright*right
    flat_p = sources.bix*n+sources.modes[:,0]
    flat_q = sources.bix*n+sources.modes[:,1]
    if two.partition == "balanced_log":
        ell_p,ell_q = ell.reshape(-1)[flat_p],ell.reshape(-1)[flat_q]
        common = .5*(gleft*left+gright*right)
        gell = _interleaved((flat_p,flat_q),(common/ell_p,common/ell_q),size=r*n)
    else:
        # Direct derivatives also cover a simple zero eigenvalue.
        half_log_tau = .5*math.log(two.tau2)
        t = np.asarray(two.t,dtype=np.float64)
        gell = _interleaved((flat_p,flat_q),(gleft*np.exp(half_log_tau+t),-(gright*np.exp(half_log_tau-t))),
                            size=r*n)
    return gell.reshape(r,n),gt


def _normalized_state(two,bounds,options):
    widths = bounds[:,1]-bounds[:,0]
    if np.any(widths <= options.bound_switch_atol):
        raise GaugeRegularityError("Collapsed/near-collapsed partition interval: "
                                   "a two-sided fixed-domain gauge derivative is not reliable")
    if np.any(two.t < bounds[:,0]) or np.any(two.t > bounds[:,1]):
        raise ValueError("Partitions lie outside the requested closed bounds")
    return 2*(two.t-bounds[:,0])/widths-1


def _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options):
    """Eigenvalue adjoint of gt*(c'(O)+z*h'(O)), t=c+h*z, for every source at once.

    Interval ends that are the amplitude bound move with the eigenvalues; ends
    at the t cap do not. An end within bound_switch_atol of the cap is a tie
    (GaugeRegularityError, first source reported, lower end before upper end).
    """
    rho,cap = _partition_limits(two.amplitude_bound,amplitude_limit,max_abs_t)
    ell = np.asarray(two.gauge.eigenvalues,dtype=np.float64)
    r,n = ell.shape
    sources = two.sources
    lower_amp,upper_amp = interval_logs(ell,sources,two.tau2,two.partition,math.log(rho))
    lower_weight,upper_weight = gt*(1-z)/2,gt*(1+z)/2
    lower_on,upper_on = lower_weight != 0,upper_weight != 0
    lower_tie = lower_on & (np.abs(lower_amp+cap) <= options.bound_switch_atol)
    upper_tie = upper_on & (np.abs(upper_amp-cap) <= options.bound_switch_atol)
    if np.any(lower_tie | upper_tie):
        i = int(np.flatnonzero(lower_tie | upper_tie)[0])
        end = "lower" if lower_tie[i] else "upper"
        raise GaugeRegularityError(f"Source {i}: {end} amplitude bound ties the finite t cap")
    lower_on &= lower_amp > -cap
    upper_on &= upper_amp < cap
    left,right = source_eigenvalues(ell,sources)
    flat_p = sources.bix*n+sources.modes[:,0]
    flat_q = sources.bix*n+sources.modes[:,1]
    with np.errstate(divide="ignore",invalid="ignore"):
        if two.partition == "balanced_log":
            result = _interleaved(
                (flat_p,flat_q,flat_p,flat_q),
                (.5*lower_weight/left,.5*lower_weight/right,-(.5*upper_weight/left),-(.5*upper_weight/right)),
                (lower_on,lower_on,upper_on,upper_on),r*n)
        else:
            result = _interleaved((flat_q,flat_p),(lower_weight/right,-(upper_weight/left)),
                                  (lower_on,upper_on),r*n)
    return result.reshape(r,n)


def _assemble(problem,decomposition,options,mode,*,bounds=None,amplitude_limit=None,
              max_abs_t=30.,inner_result=None,loss_kind="weighted",check=True,regular=False,
              space=None):
    """Loss and gauge derivative; check=False trusts a GaugeEvaluator decomposition.

    regular=True means check_gauge_regularity already passed at this gauge
    (GaugeEvaluator runs it before any inner work); otherwise the zero-eigenvalue
    and eigengap tests run here, in the order the per-factor code used.
    """
    if check:
        _validate_state(problem,decomposition)
    two = decomposition.two_body
    loss,gamp,gX = _loss_pullback(decomposition,loss_kind,check,space)
    gell,gt = _amplitude_pullback(two,gamp,options,checked=regular)
    extra = np.zeros_like(gell)
    z = None
    if bounds is not None:
        z = _normalized_state(two,bounds,options)
        extra = _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options)
    ell,X = two.gauge.eigenvalues,two.gauge.X
    gaps = _eigengaps(ell)
    if not regular:
        _check_eigengaps(ell,options,gaps)
    gL = _eigh_pullback(ell,X,gell,gX,gaps)
    extraL = (X*extra[:,None,:])@np.swapaxes(X,1,2)
    original = problem.data.chol.reshape(problem.data.nchol,-1)
    fixed = gL.reshape(problem.data.nchol,-1)@original.T
    correction = extraL.reshape(problem.data.nchol,-1)@original.T
    ambient = fixed+correction
    if not all(np.isfinite(x).all() for x in (gL,gell,gt,ambient)):
        raise FloatingPointError("Nonfinite gauge gradient")
    local = ambient@two.gauge.O.T
    generator = (local-local.T)/2
    tangent = generator@two.gauge.O
    return GaugeGradient(
        float(loss),decomposition,mode,_readonly(ambient),_readonly(tangent),
        _readonly(generator),_readonly(fixed),_readonly(correction),_readonly(gt),
        _readonly(gell+extra),inner_result=inner_result)


def fixed_partition_gradient(problem,decomposition,*,options=None,loss_kind="weighted"):
    """d L/dO with physical log ratios t (and the shift) held fixed, including eigh.

    loss_kind selects the unnormalized L_w ('weighted', default) or L_raw. No partition-bound
    response is part of this derivative. Trial gauges with the same t must
    independently satisfy the open amplitude domain. The one-body channels
    do not depend on O.
    """
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_t",
                     loss_kind=loss_kind_name(loss_kind))


def normalized_partition_gradient(problem,decomposition,*,amplitude_limit=None,
                                  max_abs_t=30.,options=None,loss_kind="weighted"):
    """Gauge derivative at fixed z in t=(lower+upper)/2+(upper-lower)*z/2.

    This supports both maps and capped/asymmetric intervals. It is different
    from a fixed-t derivative even when no partition optimizer is involved.
    The shift is held fixed; its bound depends on h only, not on O.
    """
    two = decomposition.two_body
    bounds = _partition_intervals(two.gauge.eigenvalues,two.sources,two.tau2,
                                 two.partition,two.amplitude_bound,
                                 amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_z",
                     bounds=bounds,amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                     loss_kind=loss_kind_name(loss_kind))


def _relaxed_gradient(problem,result,options,regular=False,space=None):
    """Envelope gradient of a converged inner solve on a trusted decomposition.

    No consistency re-checks: the inner solver has just verified its own
    stopping test. relaxed_gauge_gradient adds them for supplied results.
    regular=True: check_gauge_regularity already passed at this gauge; space:
    the ModeSpace of the result's bases, if already built.
    """
    if not result.converged:
        raise InnerConvergenceError("Relaxed gradient requires a converged partition solve",result)
    try:
        derivative = _assemble(problem,result.decomposition,options,"fixed_z",
                               bounds=result.bounds,amplitude_limit=result.amplitude_limit,
                               max_abs_t=result.max_abs_t,loss_kind=result.loss_kind,check=False,
                               regular=regular,space=space)
    except (GaugeRegularityError,FloatingPointError,np.linalg.LinAlgError) as exc:
        # A rejected outer trial can still account for the completed inner work.
        exc.inner_result = result
        raise
    return replace(derivative,mode="relaxed",inner_result=result)


def relaxed_gauge_gradient(problem,result,*,options=None):
    """Envelope gradient at a converged step-4 solution, including moving bounds.

    Uses the recorded amplitude_limit/max_abs_t, not guessed default bounds.
    Assumes a regular consistently tracked local inner minimum; no claim of
    a global inner solution is made for a general multi-source problem.
    """
    if not isinstance(result,PartitionOptimizationResult):
        raise TypeError("result must be PartitionOptimizationResult")
    if not result.converged:
        raise InnerConvergenceError("Relaxed gradient requires a converged partition solve",result)
    try:
        derivative = normalized_partition_gradient(
            problem,result.decomposition,amplitude_limit=result.amplitude_limit,
            max_abs_t=result.max_abs_t,options=options,loss_kind=result.loss_kind)
    except (GaugeRegularityError,FloatingPointError,np.linalg.LinAlgError) as exc:
        # A rejected outer trial can still account for the completed inner work.
        exc.inner_result = result
        raise
    two = result.decomposition.two_body
    bounds = _partition_intervals(two.gauge.eigenvalues,two.sources,two.tau2,
                                 two.partition,two.amplitude_bound,
                                 amplitude_limit=result.amplitude_limit,max_abs_t=result.max_abs_t)
    if not np.array_equal(bounds,result.bounds):
        raise ValueError("Stored inner bounds disagree with the recorded bound settings")
    residual = projected_partition_gradient(two.t,derivative.partition_gradient,bounds)
    norm = np.max(np.abs(residual),initial=0.)
    tolerance = result.gradient_threshold+128*np.finfo(float).eps*max(
        1.,np.max(np.abs(derivative.partition_gradient),initial=0.))
    if norm > tolerance:
        raise InnerConvergenceError("Stored solution fails a fresh partition KKT check",result)
    if not np.isclose(derivative.loss,result.loss,rtol=1e-11,atol=1e-14):
        raise ValueError("Stored inner loss does not match its decomposition")
    if result.optimize_shift:
        # Envelope over mu: mu must be the exact (clipped) minimizer at the stored t.
        # Its bound depends on h only, so no moving-bound term arises for mu.
        tolerance = result.shift_tolerance+128*np.finfo(float).eps*max(
            1.,float(np.max(np.abs(result.decomposition.one_body.eigenvalues),initial=0.)))
        for profile in shift_coefficients(result.decomposition,loss_kind=result.loss_kind,
                                          one_body_limit=result.one_body_limit):
            if abs(profile.mu-profile.update) > tolerance:
                raise InnerConvergenceError("Stored shift is not the exact minimizer at the stored t",result)
    return replace(derivative,mode="relaxed",inner_result=result)


class GaugeEvaluator:
    """Reusable step-5 evaluator: exact tensors are validated before construction.

    Cache fixed h channels/source topology, rotate/eigh once at each trial O,
    then evaluate without n^4 ERI reconstruction. No outer optimizer is run.
    loss_kind='weighted' (default) or 'raw' selects the objective for every
    method. shift=(mu_up, mu_down) (default zero) is the fixed one-body shift;
    a nonzero shift requires problem.nelec.
    Select derivative computation with backend='numpy' (default). The reserved
    backend='pytorch' raises before initialization performs numerical work.
    fixed, fixed_z and relaxed run check_gauge_regularity right after the
    trial gauge is diagonalized, so near-degenerate factor eigenvalues (and,
    for balanced_log, near-zero ones) raise GaugeRegularityError before any
    inner sweep or loss evaluation.
    """

    def __init__(self,problem,*,tau1,tau2,partition="balanced_log",amplitude_bound=1.,
                 options=None,backend="numpy",loss_kind="weighted",shift=None):
        self._backend = _resolve_derivative_backend(backend)
        if not isinstance(problem,PreparedProblem):
            raise TypeError("problem must be a PreparedProblem")
        self.problem = problem
        self.tau2 = positive_scale(tau2,"tau2")
        self.amplitude_bound = positive_scale(amplitude_bound,"amplitude_bound")
        if self.amplitude_bound > 1 or partition not in ("balanced_log","anchored_log"):
            raise ValueError("Expected a supported partition map and amplitude_bound<=1")
        self.partition = partition
        self.loss_kind = loss_kind_name(loss_kind)
        self.options = _gradient_options(options)
        self.sources = enumerate_sources(problem)
        # Default one-body channels at the engine's fixed shift (also the start for relaxation).
        self.one_body = build_one_body_channels(problem,tau1,shift_pair(shift))

    @property
    def backend(self):
        """Derivative backend chosen at construction (read-only)."""
        return self._backend

    @property
    def shift(self):
        """Fixed (mu_up, mu_down) used unless a call supplies or relaxes one."""
        return self.one_body.shift

    def _one_body(self,shift=None):
        if shift is None:
            return self.one_body
        return one_body_at_shift(self.one_body,shift_pair(shift),self.problem.nelec)

    def _gauge(self,O):
        """Rotate and diagonalize; only a caller's O is checked (orthogonality)."""
        data = self.problem.data
        O = np.eye(data.nchol) if O is None else _gauge_array(O,data.nchol,self.problem.report.options)
        chol = (O@data.chol.reshape(data.nchol,-1)).reshape(data.chol.shape)
        ell,X = np.linalg.eigh(chol)
        return GaugeState(_readonly(O),_readonly(chol),_readonly(ell),_readonly(X))

    def _regular_gauge(self,O):
        """_gauge plus check_gauge_regularity: a singular trial fails before any inner work."""
        gauge = self._gauge(O)
        check_gauge_regularity(gauge,self.sources,self.partition,self.options)
        return gauge

    def _decomposition(self,gauge,t,shift=None):
        t,d = fixed_partition_amplitudes(gauge.eigenvalues,self.sources,self.tau2,t,
                                         self.partition,self.amplitude_bound)
        two = _two_body_from_amplitudes(gauge,self.problem.policy,self.problem.nelec,self.tau2,
                                        self.partition,self.amplitude_bound,self.sources,t,d)
        return FixedDecomposition(self._one_body(shift),two)

    def decomposition(self,O=None,*,t=0.,shift=None):
        """Fast channel construction for a trial orthogonal gauge, with fixed t and shift."""
        return self._decomposition(self._gauge(O),t,shift)

    def fixed(self,O=None,*,t=0.,shift=None):
        """Loss and derivative with t (and the shift) fixed."""
        return _assemble(self.problem,self._decomposition(self._regular_gauge(O),t,shift),self.options,
                         "fixed_t",loss_kind=self.loss_kind,check=False,regular=True)

    def _bounds(self,gauge,amplitude_limit,max_abs_t):
        return _partition_intervals(gauge.eigenvalues,self.sources,self.tau2,
                                    self.partition,self.amplitude_bound,
                                    amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)

    def fixed_z(self,O=None,*,z=0.,amplitude_limit=None,max_abs_t=30.,shift=None):
        """Evaluate a gauge at fixed normalized coordinates z in [-1,1] and a fixed shift."""
        gauge = self._regular_gauge(O)
        bounds = self._bounds(gauge,amplitude_limit,max_abs_t)
        z = _coordinates(z,len(self.sources),"z")
        if np.any(np.abs(z)>1):
            raise ValueError("z must lie in [-1,1]")
        t = bounds[:,0]+(z+1)*(bounds[:,1]-bounds[:,0])/2
        t = np.where(z == 1,bounds[:,1],np.where(z == -1,bounds[:,0],t))
        return _assemble(self.problem,self._decomposition(gauge,t,shift),self.options,"fixed_z",
                         bounds=bounds,amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                         loss_kind=self.loss_kind,check=False,regular=True)

    def relaxed(self,O=None,*,t0=None,options=None):
        """Relax partitions (and optionally the shift) at THIS gauge; envelope gradient.

        options: InnerOptions (defaults when None), the inner-solver settings.
        Warm-start log ratios are projected into the current intervals. With
        t0=None, use interval midpoints. With options.optimize_shift=True the
        shift starts at mu*(t0) (no shift seed) and is relaxed together with t;
        the engine shift is used only if it is feasible, otherwise the lower
        bound. Every call, including line-search or finite-difference trial
        gauges, performs its own converged inner solve.
        """
        options = _inner_options(options)
        gauge = self._regular_gauge(O)
        bounds = self._bounds(gauge,options.amplitude_limit,options.max_abs_t)
        t = bounds.mean(axis=1) if t0 is None else np.clip(
            _coordinates(t0,len(self.sources),"t0"),bounds[:,0],bounds[:,1])
        shift = None
        if options.optimize_shift:
            lower = shift_bounds(self.one_body,options.one_body_limit)
            shift = tuple(mu if low is None else max(mu,low)
                          for mu,low in zip(self.one_body.shift,lower))
        # One mode space per trial gauge: the inner solve and the envelope gradient share it.
        space = ModeSpace(gauge.X,self.one_body.X)
        inner = _solve_partitions(self._decomposition(gauge,t,shift),options,self.loss_kind,space)
        return _relaxed_gradient(self.problem,inner,self.options,regular=True,space=space)


def check_gauge_direction(evaluator,O,K,*,steps=(1e-3,3e-4,1e-4,3e-5,1e-5)):
    """Compare dF[KO] against central differences along exp(h*K)@O.

    evaluator must return GaugeGradient and use the SAME fixed-t, fixed-z or
    relaxed definition at every point. Passing a relaxed callable re-solves
    its inner problem at every positive/negative trial point.
    """
    O,K = _real_array(O,"O"),_real_array(K,"K")
    if O.ndim != 2 or O.shape[0] != O.shape[1] or K.shape != O.shape:
        raise ValueError("O,K must be square matrices of the same shape")
    if not np.allclose(K+K.T,0.,atol=1e-12,rtol=0):
        raise ValueError("K must be skew-symmetric")
    center = evaluator(O)
    if not isinstance(center,GaugeGradient):
        raise TypeError("evaluator must return GaugeGradient")
    analytic = float(np.sum(center.gradient_O*(K@O)))
    differences = []
    for step in steps:
        step = positive_scale(step,"step")
        plus = evaluator(expm(step*K)@O)
        minus = evaluator(expm(-step*K)@O)
        finite = (plus.loss-minus.loss)/(2*step)
        error = abs(finite-analytic)
        scale = max(abs(finite),abs(analytic),np.finfo(float).tiny)
        differences.append(GaugeDifference(step,float(finite),float(error),float(error/scale)))
    return GaugeGradientCheck(analytic,tuple(differences))
