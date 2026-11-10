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
    TwoBodyDecomposition,
)
from .loss import commutator_norm2, OverlapEvaluator, loss_kind_name
from .partitions import (
    _readonly,
    _partition_intervals,
    _partition_limits,
    _source_channels,
    _validate_partition_state,
    optimize_partitions,
    projected_partition_gradient,
)
from .problem import (
    exponential_gauge,
    givens_gauge,
    skew_from_parameters,
    _real_array,
    diagonalize_factors,
    rotate_cholesky,
    _gauge_array,
)
from dataclasses import replace
from .channels import (
    build_one_body_channels,
    enumerate_sources,
    fixed_partition_amplitudes,
    one_body_at_shift,
    positive_scale,
    shift_pair,
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


def commutator_norm2_derivatives(d, e, overlap):
    """Return (norm2, d_norm2/dd, d_norm2/de, d_norm2/dC).

    Differentiate the polynomial norm formula, not its roundoff clamp. The
    overlap pullback must include BOTH bases. Blocks may contain zero legs.
    """
    d, e, C = (_real_array(x, n) for x, n in ((d,"d"),(e,"e"),(overlap,"overlap")))
    value = commutator_norm2(d, e, C)
    R = (C*e[None,:]) @ C.T
    S = C.T @ (d[:,None]*C)
    with np.errstate(over="raise", invalid="raise"):
        gd = 4*d*((C*C) @ (e*e))-4*((R*R) @ d)
        ge = 4*e*((C*C).T @ (d*d))-4*((S*S) @ e)
        gC = 4*(d*d)[:,None]*C*(e*e)[None,:]
        gC -= 8*((d[:,None]*R)*d[None,:]) @ (C*e[None,:])
    if not all(np.isfinite(x).all() for x in (gd,ge,gC)):
        raise FloatingPointError("Nonfinite overlap-norm derivative")
    return value, _readonly(gd), _readonly(ge), _readonly(gC)


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
    gap = ell[None,:]-ell[:,None]
    mask = ~np.eye(ell.size,dtype=bool)
    tolerance = options.gap_atol+options.gap_rtol*np.max(np.abs(ell),initial=0.)
    if np.any(np.abs(gap[mask]) <= tolerance):
        raise GaugeRegularityError("Repeated or near-degenerate Cholesky eigenvalues; "
                                   "the simple-eigenpair gradient is not defined reliably")
    inverse = np.zeros_like(gap)
    np.divide(1.,gap,out=inverse,where=mask)
    term = (X.T@gX)*inverse
    result = X@(np.diag(ge)+(term+term.T)/2)@X.T
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite symmetric-eigh derivative")
    return _readonly((result+result.T)/2)


def exponential_gauge_vjp(theta, nchol, gradient_O, reference=None, *, backend="numpy"):
    """Gradient in the packed global exponential coordinates of prepare.py.

    O=expm(S(theta))@reference. Uses the exact Frechet adjoint; the local
    skew generator gradient is not the global theta gradient away from zero.
    The optional backend matches GaugeEvaluator; 'pytorch' is not implemented.
    """
    _resolve_derivative_backend(backend)
    O = exponential_gauge(theta,nchol,reference)
    E = _real_array(gradient_O,"gradient_O")
    if E.shape != O.shape:
        raise ValueError("gradient_O must have shape (nchol,nchol)")
    ref = np.eye(nchol) if reference is None else np.asarray(reference,dtype=float)
    S = skew_from_parameters(theta,nchol)
    adjoint = expm_frechet(S.T,E@ref.T,compute_expm=False)
    rows, cols = np.triu_indices(nchol,1)
    return _readonly(adjoint[rows,cols]-adjoint[cols,rows])


def givens_gauge_vjp(nchol, rotations, gradient_O, reference=None, *, backend="numpy"):
    """Angle gradient in the exact ordered Givens sequence of prepare.py.

    The optional backend matches GaugeEvaluator; 'pytorch' is not implemented.
    """
    _resolve_derivative_backend(backend)
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


def _loss_pullback(decomposition, loss_kind="raw"):
    """Return the loss_kind loss, actual-channel leg adjoints and basis adjoints.

    Keep zero legs in each block: a zero leg in a nonzero same-spin block can
    have a nonzero amplitude derivative. Identity-only pairs need no special
    case beyond the polynomial kernel. For loss_kind='weighted' the legs are
    scaled by sqrt(a_i) and the leg adjoints are chained back to the actual d.
    """
    evaluator = OverlapEvaluator(decomposition, loss_kind=loss_kind)
    root = evaluator.root_scales
    two = decomposition.two_body
    offset = len(decomposition.one_body.channels)
    blocks = []
    for index,channel in enumerate(decomposition.one_body.channels):
        pair = [None,None]
        pair[channel.spin] = (-1,(channel.mode,),np.array([root[index]*channel.d]),())
        blocks.append(tuple(pair))
    for index,channel in enumerate(two.channels):
        spin_legs = [[],[]]
        scale = root[offset+index]
        for leg,(spin,mode,d) in enumerate(zip(channel.spins,channel.modes,channel.d)):
            spin_legs[spin].append((mode,scale*d,leg))
        blocks.append(tuple(None if not legs else (
            channel.bix,tuple(x[0] for x in legs),np.array([x[1] for x in legs]),
            tuple((index,x[2]) for x in legs)) for legs in spin_legs))
    g_amplitude = np.zeros((len(two.channels),2))
    gX = np.zeros_like(two.gauge.X)
    terms = []
    for i,left_pair in enumerate(blocks):
        for right_pair in blocks[i+1:]:
            for left,right in zip(left_pair,right_pair):
                if left is None or right is None or left[0] == right[0]:
                    continue  # Shared-basis diagonal blocks commute for every gauge.
                a,ma,d,legs_a = left
                b,mb,e,legs_b = right
                C = evaluator._overlap(a,b)[np.ix_(ma,mb)]
                value,gd,ge,gC = commutator_norm2_derivatives(d,e,C)
                terms.append(value)
                for key,value in zip(legs_a,gd):
                    g_amplitude[key] += value
                for key,value in zip(legs_b,ge):
                    g_amplitude[key] += value
                if a >= 0:
                    gX[a][:,ma] += evaluator._bases[b][:,mb]@gC.T
                if b >= 0:
                    gX[b][:,mb] += evaluator._bases[a][:,ma]@gC
    # d(scaled leg)/d(actual leg) = sqrt(a_i); exactly one for loss_kind='raw'.
    if evaluator.loss_kind != "raw":
        g_amplitude *= root[offset:,None]
    return math.fsum(terms),g_amplitude,gX


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


def _amplitude_pullback(two,g_amplitude,options):
    gell = np.zeros_like(two.gauge.eigenvalues)
    gt = np.zeros(len(two.sources))
    for i,(source,(start,stop)) in enumerate(zip(two.sources,two.source_slices)):
        signs = np.array((1.,-1.) if source.kind == "two_branch" else (1.,-1.,-1.,1.))
        gleft,gright = np.sum(signs[:,None]*g_amplitude[start:stop],axis=0)
        left,right = two.amplitudes[i]
        gt[i] = gleft*left-gright*right
        g,(p,q) = source.bix,source.modes
        ell = two.gauge.eigenvalues[g]
        if two.partition == "balanced_log":
            tolerance = options.zero_atol+options.zero_rtol*np.max(np.abs(ell),initial=0.)
            if abs(ell[p]) <= tolerance or abs(ell[q]) <= tolerance:
                raise GaugeRegularityError(
                    f"Source {i}, factor {g}: balanced amplitudes meet a zero/near-zero "
                    "eigenvalue; a smooth sign branch is required")
            common = .5*(gleft*left+gright*right)
            gell[g,p] += common/ell[p]
            gell[g,q] += common/ell[q]
        else:
            # Direct derivatives also cover a simple zero eigenvalue.
            gell[g,p] += gleft*math.exp(.5*math.log(two.tau2)+two.t[i])
            gell[g,q] -= gright*math.exp(.5*math.log(two.tau2)-two.t[i])
    return gell,gt


def _normalized_state(two,bounds,options):
    widths = bounds[:,1]-bounds[:,0]
    if np.any(widths <= options.bound_switch_atol):
        raise GaugeRegularityError("Collapsed/near-collapsed partition interval: "
                                   "a two-sided fixed-domain gauge derivative is not reliable")
    if np.any(two.t < bounds[:,0]) or np.any(two.t > bounds[:,1]):
        raise ValueError("Partitions lie outside the requested closed bounds")
    return 2*(two.t-bounds[:,0])/widths-1


def _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options):
    """Eigenvalue adjoint of gt*(c'(O)+z*h'(O)), t=c+h*z."""
    rho,cap = _partition_limits(two.amplitude_bound,amplitude_limit,max_abs_t)
    log_rho = math.log(rho)
    result = np.zeros_like(two.gauge.eigenvalues)
    for i,source in enumerate(two.sources):
        g,(p,q) = source.bix,source.modes
        left,right = two.gauge.eigenvalues[g,p],two.gauge.eigenvalues[g,q]
        if two.partition == "balanced_log":
            base = .5*(math.log(abs(left))+math.log(abs(right))+math.log(two.tau2))
            log_left = log_right = base
        else:
            log_left = -np.inf if left == 0 else math.log(abs(left))+.5*math.log(two.tau2)
            log_right = -np.inf if right == 0 else math.log(abs(right))+.5*math.log(two.tau2)
        lower_amp,upper_amp = log_right-log_rho,log_rho-log_left
        lower_weight,upper_weight = gt[i]*(1-z[i])/2,gt[i]*(1+z[i])/2
        if lower_weight != 0:
            if abs(lower_amp+cap) <= options.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: lower amplitude bound ties the finite t cap")
            if lower_amp > -cap:
                if two.partition == "balanced_log":
                    result[g,p] += .5*lower_weight/left
                    result[g,q] += .5*lower_weight/right
                else:
                    result[g,q] += lower_weight/right
        if upper_weight != 0:
            if abs(upper_amp-cap) <= options.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: upper amplitude bound ties the finite t cap")
            if upper_amp < cap:
                if two.partition == "balanced_log":
                    result[g,p] -= .5*upper_weight/left
                    result[g,q] -= .5*upper_weight/right
                else:
                    result[g,p] -= upper_weight/left
    return result


def _assemble(problem,decomposition,options,mode,*,bounds=None,amplitude_limit=None,
              max_abs_t=30.,inner_result=None,loss_kind="raw"):
    _validate_state(problem,decomposition)
    two = decomposition.two_body
    loss,gamp,gX = _loss_pullback(decomposition,loss_kind)
    gell,gt = _amplitude_pullback(two,gamp,options)
    extra = np.zeros_like(gell)
    z = None
    if bounds is not None:
        z = _normalized_state(two,bounds,options)
        extra = _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options)
    gL,extraL = [],[]
    for g in range(len(gell)):
        try:
            gL.append(eigh_vjp(two.gauge.eigenvalues[g],two.gauge.X[g],gell[g],gX[g],options=options))
        except GaugeRegularityError as exc:
            raise GaugeRegularityError(f"Factor {g}: {exc}") from exc
        extraL.append((two.gauge.X[g]*extra[g])@two.gauge.X[g].T)
    gL,extraL = np.array(gL),np.array(extraL)
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
        _readonly(gell+extra),_readonly(gX),_readonly(gL+extraL),
        None if z is None else _readonly(z),inner_result)


def fixed_partition_gradient(problem,decomposition,*,options=None,backend="numpy",
                             loss_kind="raw"):
    """d L/dO with physical log ratios t (and the shift) held fixed, including eigh.

    loss_kind selects L_raw or the unnormalized L_w. No partition-bound
    response is part of this derivative. Trial gauges with the same t must
    independently satisfy the open amplitude domain. The one-body channels
    do not depend on O. backend='pytorch' is reserved and raises.
    """
    _resolve_derivative_backend(backend)
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_t",
                     loss_kind=loss_kind_name(loss_kind))


def normalized_partition_gradient(problem,decomposition,*,amplitude_limit=None,
                                  max_abs_t=30.,options=None,backend="numpy",loss_kind="raw"):
    """Gauge derivative at fixed z in t=(lower+upper)/2+(upper-lower)*z/2.

    This supports both maps and capped/asymmetric intervals. It is different
    from a fixed-t derivative even when no partition optimizer is involved.
    The shift is held fixed; its bound depends on h only, not on O.
    backend='pytorch' is reserved and raises NotImplementedError.
    """
    _resolve_derivative_backend(backend)
    two = decomposition.two_body
    bounds = _partition_intervals(two.gauge.eigenvalues,two.sources,two.tau2,
                                 two.partition,two.amplitude_bound,
                                 amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_z",
                     bounds=bounds,amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                     loss_kind=loss_kind_name(loss_kind))


def relaxed_gauge_gradient(problem,result,*,options=None,backend="numpy"):
    """Envelope gradient at a converged step-4 solution, including moving bounds.

    Uses the recorded amplitude_limit/max_abs_t, not guessed default bounds.
    Assumes a regular consistently tracked local inner minimum; no claim of
    a global inner solution is made for a general multi-source problem.
    backend='pytorch' is rejected before inspecting the inner result.
    """
    backend = _resolve_derivative_backend(backend)
    if not isinstance(result,PartitionOptimizationResult):
        raise TypeError("result must be PartitionOptimizationResult")
    if not result.converged:
        raise InnerConvergenceError("Relaxed gradient requires a converged partition solve",result)
    try:
        derivative = normalized_partition_gradient(
            problem,result.decomposition,amplitude_limit=result.amplitude_limit,
            max_abs_t=result.max_abs_t,options=options,backend=backend,
            loss_kind=result.loss_kind)
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
    loss_kind='raw' (default) or 'weighted' selects the objective for every
    method. shift=(mu_up, mu_down) (default zero) is the fixed one-body shift;
    a nonzero shift requires problem.nelec.
    Select derivative computation with backend='numpy' (default). The reserved
    backend='pytorch' raises before initialization performs numerical work.
    """

    def __init__(self,problem,*,tau1,tau2,partition="balanced_log",amplitude_bound=1.,
                 options=None,backend="numpy",loss_kind="raw",shift=None):
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
        counts = [2 if s.kind == "two_branch" else 4 for s in self.sources]
        endpoints = np.r_[0,np.cumsum(counts)]
        self.source_slices = tuple((int(a),int(b)) for a,b in zip(endpoints[:-1],endpoints[1:]))

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
        data = self.problem.data
        O = np.eye(data.nchol) if O is None else O
        O = _gauge_array(O,data.nchol,self.problem.report.options)
        chol = rotate_cholesky(data.chol,O,self.problem.report.options)
        ell,X = diagonalize_factors(chol,self.problem.report.options)
        return GaugeState(_readonly(O),_readonly(chol),_readonly(ell),_readonly(X))

    def _decomposition(self,gauge,t,shift=None):
        t,d = fixed_partition_amplitudes(gauge.eigenvalues,self.sources,self.tau2,t,
                                         self.partition,self.amplitude_bound)
        channels = tuple(c for source,amplitude in zip(self.sources,d)
                         for c in _source_channels(source,amplitude,self.tau2))
        two = TwoBodyDecomposition(gauge,self.problem.policy,self.problem.nelec,self.tau2,
                                   self.partition,self.amplitude_bound,self.sources,t,d,
                                   self.source_slices,channels)
        return FixedDecomposition(self._one_body(shift),two)

    def decomposition(self,O=None,*,t=0.,shift=None):
        """Fast channel construction for a trial orthogonal gauge, with fixed t and shift."""
        return self._decomposition(self._gauge(O),t,shift)

    def fixed(self,O=None,*,t=0.,shift=None):
        """Loss and derivative with t (and the shift) fixed."""
        return fixed_partition_gradient(self.problem,self.decomposition(O,t=t,shift=shift),
                                         options=self.options,backend=self.backend,
                                         loss_kind=self.loss_kind)

    def _bounds(self,gauge,amplitude_limit,max_abs_t):
        return _partition_intervals(gauge.eigenvalues,self.sources,self.tau2,
                                    self.partition,self.amplitude_bound,
                                    amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)

    def fixed_z(self,O=None,*,z=0.,amplitude_limit=None,max_abs_t=30.,shift=None):
        """Evaluate a gauge at fixed normalized coordinates z in [-1,1] and a fixed shift."""
        gauge = self._gauge(O)
        bounds = self._bounds(gauge,amplitude_limit,max_abs_t)
        z = _coordinates(z,len(self.sources),"z")
        if np.any(np.abs(z)>1):
            raise ValueError("z must lie in [-1,1]")
        t = bounds[:,0]+(z+1)*(bounds[:,1]-bounds[:,0])/2
        t = np.where(z == 1,bounds[:,1],np.where(z == -1,bounds[:,0],t))
        return normalized_partition_gradient(self.problem,self._decomposition(gauge,t,shift),
                                              amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                                              options=self.options,backend=self.backend,
                                              loss_kind=self.loss_kind)

    def relaxed(self,O=None,*,t0=None,max_sweeps=500,gradient_atol=1e-11,
                gradient_rtol=1e-9,amplitude_limit=None,max_abs_t=30.,
                optimize_shift=False,one_body_limit=None,shift_atol=0.,shift_rtol=1e-8):
        """Relax partitions (and optionally the shift) at THIS gauge; envelope gradient.

        Warm-start log ratios are projected into the current intervals. With
        t0=None, use interval midpoints. With optimize_shift=True the shift
        starts at mu*(t0) (no shift seed) and is relaxed together with t; the
        engine shift is used only if it is feasible, otherwise the lower bound.
        Every call, including line-search or finite-difference trial gauges,
        performs its own converged inner solve.
        """
        gauge = self._gauge(O)
        bounds = self._bounds(gauge,amplitude_limit,max_abs_t)
        t = bounds.mean(axis=1) if t0 is None else np.clip(
            _coordinates(t0,len(self.sources),"t0"),bounds[:,0],bounds[:,1])
        shift = None
        if optimize_shift:
            lower = shift_bounds(self.one_body,one_body_limit)
            shift = tuple(mu if low is None else max(mu,low)
                          for mu,low in zip(self.one_body.shift,lower))
        inner = optimize_partitions(self._decomposition(gauge,t,shift),max_sweeps=max_sweeps,
                                    gradient_atol=gradient_atol,gradient_rtol=gradient_rtol,
                                    amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                                    loss_kind=self.loss_kind,optimize_shift=optimize_shift,
                                    one_body_limit=one_body_limit,shift_atol=shift_atol,
                                    shift_rtol=shift_rtol)
        return relaxed_gauge_gradient(self.problem,inner,options=self.options,backend=self.backend)


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
