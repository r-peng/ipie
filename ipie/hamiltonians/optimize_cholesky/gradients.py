"""Analytic gauge gradients and explicit derivative-backend selection."""

import copy
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
    _log_leg_scale,
    _one_body_from_eigensystem,
    _two_body_from_amplitudes,
    build_one_body_channels,
    check_cholesky_shift,
    cholesky_shift_array,
    enumerate_sources,
    fixed_partition_amplitudes,
    leg_log_bases,
    leg_usage,
    leg_values,
    map_eigenvalues,
    one_body_at_shift,
    one_body_eigensystem,
    one_body_matrices,
    positive_scale,
    shift_pair,
    source_eigenvalues,
    split_cholesky_shift,
)
from .shift import check_one_body_shift, one_body_limit_value, shift_bounds, shift_coefficients


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


def _cholesky_gradient_option(kind):
    """Validate a cholesky_gradient request: None, 'tied' or 'split'."""
    if kind is None or (isinstance(kind,str) and kind in ("tied","split")):
        return kind
    raise ValueError("cholesky_gradient must be None, 'tied' or 'split'")


def _cholesky_gradient_kind(kind,two):
    """The Cholesky-shift gradient to report for two: None, 'tied' or 'split'.

    None picks the shift's own form: no gradient without a shift; 'tied' when
    every spin that carries channels has the same shift (always for aa_only,
    whose mu_down is 0); 'split' otherwise.
    """
    kind = _cholesky_gradient_option(kind)
    if kind is not None:
        return kind
    mu = two.cholesky_shift
    if mu is None:
        return None
    return "split" if two.policy != "aa_only" and split_cholesky_shift(mu) else "tied"


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
    if two.cholesky_shift is not None:
        _check_one_body_system(problem,decomposition)


def _check_one_body_system(problem,decomposition):
    """The stored one-body eigenpairs must reconstruct h_sigma at this gauge and Cholesky shift.

    Without a shift that is the bare h. The gradient differentiates this basis
    whenever a shift, or a shift gradient, is involved.
    """
    one,two = decomposition.one_body,decomposition.two_body
    h = problem.data.h1
    mu = two.cholesky_shift
    matrices = (h,h) if mu is None else one_body_matrices(h,two.gauge.chol,mu,two.nelec)
    atol,rtol = problem.report.options.atol,problem.report.options.rtol
    n = problem.data.nbasis
    for s in one.spins:
        X,eps = np.asarray(one.spin_X(s)),np.asarray(one.spin_eigenvalues(s))
        if (X.shape != (n,n) or eps.shape != (n,)
                or not np.allclose((X*eps)@X.T,matrices[s],atol=atol,rtol=rtol)):
            raise ValueError(f"Stored one-body eigenpairs do not reconstruct h_sigma (spin {s}) "
                             "for this gauge and Cholesky shift")


def _check_zero_eigenvalues(eigenvalues,sources,partition,options):
    """balanced_log needs |ell| above zero_atol+zero_rtol*max|ell_g| on every source mode.

    Depends on the partition-map eigenvalues only, so GaugeEvaluator runs it
    before any inner work. Anchored amplitudes are differentiable at a simple
    zero. Per-spin eigenvalues (2, r, n) test each leg's scale instead
    (_check_zero_leg_scales).
    """
    if partition != "balanced_log" or len(sources) == 0:
        return
    if np.ndim(eigenvalues) == 3:
        _check_zero_leg_scales(eigenvalues,sources,options)
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


def _check_zero_leg_scales(spin_eigenvalues,sources,options):
    """Per-spin balanced map: every leg scale g above zero_atol+zero_rtol*max|lambda_g|.

    g is |lambda| for a one-spin leg and the RMS over spins for a quartet leg
    (channels.leg_log_bases); the maximum runs over both spins of factor g.
    A single spin of a quartet leg may vanish: its amplitude is smooth there.
    """
    lam = np.asarray(spin_eigenvalues,dtype=np.float64)
    scale = np.exp(_log_leg_scale(leg_values(lam,sources),leg_usage(sources)))
    tolerance = options.zero_atol+options.zero_rtol*np.max(np.abs(lam),axis=(0,2),initial=0.)
    bad = np.any(scale <= tolerance[sources.bix][:,None],axis=1)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise GaugeRegularityError(
            f"Source {i}, factor {int(sources.bix[i])}: balanced amplitudes meet a zero/near-zero "
            "eigenvalue (leg scale); a smooth sign branch is required")


def _eigengaps(eigenvalues):
    """gap[g,a,b] = ell[g,b]-ell[g,a] and the off-diagonal mask."""
    ell = np.asarray(eigenvalues,dtype=np.float64)
    return ell[:,None,:]-ell[:,:,None],~np.eye(ell.shape[1],dtype=bool)


def _degenerate_rows(eigenvalues,options,gaps=None):
    """Rows (factors or spins) with a gap |gap| <= gap_atol+gap_rtol*max|row|."""
    ell = np.asarray(eigenvalues,dtype=np.float64)
    gap,mask = _eigengaps(ell) if gaps is None else gaps
    tolerance = options.gap_atol+options.gap_rtol*np.max(np.abs(ell),axis=1,initial=0.)
    return np.any((np.abs(gap) <= tolerance[:,None,None]) & mask,axis=(1,2))


def _check_eigengaps(eigenvalues,options,gaps=None):
    """Every factor needs simple eigenvalues: |gap| > gap_atol+gap_rtol*max|ell_g|.

    The same test as eigh_vjp, for all factors at once; the first failing
    factor is reported.
    """
    bad = _degenerate_rows(eigenvalues,options,gaps)
    if np.any(bad):
        g = int(np.flatnonzero(bad)[0])
        raise GaugeRegularityError(f"Factor {g}: Repeated or near-degenerate Cholesky eigenvalues; "
                                   "the simple-eigenpair gradient is not defined reliably")


def _check_one_body_gaps(eigenvalues,options,gaps=None):
    """The one-body matrices need simple eigenvalues too, when a gradient goes through their basis.

    eigenvalues: (n,) for one shared basis or (2, n) per spin. Same tolerances
    as the factors. A degenerate h (bare h at zero shift, for example) leaves
    the loss's dependence on its basis undefined.
    """
    eps = np.asarray(eigenvalues,dtype=np.float64)
    rows = eps if eps.ndim == 2 else eps[None]
    bad = _degenerate_rows(rows,options,gaps)
    if np.any(bad):
        which = "" if eps.ndim == 1 else f" (spin {('up','down')[int(np.flatnonzero(bad)[0])]})"
        raise GaugeRegularityError(f"One-body matrix h_sigma{which}: repeated or near-degenerate "
                                   "eigenvalues; the gradient through its eigenbasis is not defined reliably")


def check_gauge_regularity(gauge,sources,partition,options=None,*,eigenvalues=None,
                           one_body_eigenvalues=None):
    """The smooth-derivative checks that depend on the gauge only, in the order _assemble applies them.

    Balanced zero eigenvalues first, then eigengaps. GaugeEvaluator calls this
    right after diagonalizing a trial gauge, so a singular trial is rejected
    before the inner solve instead of after it.

    eigenvalues: the partition-map eigenvalues for the zero test (default the
    gauge's own; with a Cholesky shift ell - mu, or (2, r, n) per spin).
    one_body_eigenvalues: (n,) or (2, n) eigenvalues of the one-body matrices,
    checked last for eigengaps when the gradient goes through their basis.
    """
    options = _gradient_options(options)
    _check_zero_eigenvalues(gauge.eigenvalues if eigenvalues is None else eigenvalues,
                            sources,partition,options)
    _check_eigengaps(gauge.eigenvalues,options)
    if one_body_eigenvalues is not None:
        _check_one_body_gaps(one_body_eigenvalues,options)


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

    ell are the partition-map eigenvalues (two.map_eigenvalues: the gauge's
    own, or ell - mu for a tied Cholesky shift). checked=True means the
    balanced zero-eigenvalue test already ran.
    """
    ell = np.asarray(two.map_eigenvalues,dtype=np.float64)
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


def _leg_targets(sources,r,n):
    """Flat index into (2, r, n) of every leg [source, leg, spin]: lambda[spin, bix, mode of the leg]."""
    return (np.arange(2)[None,None,:]*(r*n)+(sources.bix*n)[:,None,None]+sources.modes[:,:,None])


def _leg_adjoints(sources,g_amplitude):
    """(count, 2, 2) adjoints [source, leg, spin] of the per-spin leg amplitudes."""
    count = len(sources)
    signed = sources.channel_sign[:,None]*g_amplitude
    spins = sources.channel_spins
    index = tuple((sources.channel_source*2+leg)*2+spins[:,leg] for leg in (0,1))
    return _interleaved(index,(signed[:,0],signed[:,1]),size=4*count).reshape(count,2,2)


def _log_scale_gradient(values,used):
    """d log g/d lambda_s = lambda_s/sum_used lambda^2 for each leg scale g, (count, 2, 2); 0 on unused spins."""
    squares = np.sum(np.where(used,values*values,0.),axis=2,keepdims=True)
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.where(used,values/squares,0.)


def _leg_amplitude_pullback(two,g_amplitude,options,checked=False):
    """(dL/d lambda (2, r, n), dL/dt) through the per-spin map, for every source at once.

    Works on any canonical decomposition: the legs are two.leg_amplitudes and
    lambda = two.spin_eigenvalues, so a tied or absent shift gets each spin's
    derivative separately. d_L,s = lambda_L,s kappa_L e^t and d_R,s =
    -lambda_R,s kappa_R e^-t (channels.leg_log_bases). balanced_log adds
    dL/dlog g_L = -gt/2 and dL/dlog g_R = +gt/2 through the leg scales.
    checked=True means the balanced zero test already ran.
    """
    lam = np.asarray(two.spin_eigenvalues,dtype=np.float64)
    _,r,n = lam.shape
    sources = two.sources
    if not checked:
        _check_zero_eigenvalues(lam,sources,two.partition,options)
    G = _leg_adjoints(sources,g_amplitude)
    GA = G*two.leg_amplitudes
    gt = GA[:,0].sum(axis=1)-GA[:,1].sum(axis=1)
    values,used = leg_values(lam,sources),leg_usage(sources)
    t = np.asarray(two.t,dtype=np.float64)
    exponent = .5*math.log(two.tau2)+np.stack([t,-t],axis=1)
    balanced = two.partition == "balanced_log"
    if balanced:
        scale = _log_leg_scale(values,used)
        exponent = exponent+.5*(scale[:,::-1]-scale)
    with np.errstate(over="ignore"):
        factor = np.exp(exponent)*np.array([1.,-1.])          # d(leg)/d(its own lambda)
    g_leg = G*factor[:,:,None]
    if balanced:
        g_leg = g_leg+(.5*gt[:,None]*np.array([-1.,1.]))[:,:,None]*_log_scale_gradient(values,used)
    g_leg = np.where(used,g_leg,0.)
    g_lam = np.bincount(_leg_targets(sources,r,n).reshape(-1),weights=g_leg.reshape(-1),minlength=2*r*n)
    return g_lam.reshape(2,r,n),gt


def _normalized_state(two,bounds,options):
    widths = bounds[:,1]-bounds[:,0]
    if np.any(widths <= options.bound_switch_atol):
        raise GaugeRegularityError("Collapsed/near-collapsed partition interval: "
                                   "a two-sided fixed-domain gauge derivative is not reliable")
    if np.any(two.t < bounds[:,0]) or np.any(two.t > bounds[:,1]):
        raise ValueError("Partitions lie outside the requested closed bounds")
    return 2*(two.t-bounds[:,0])/widths-1


def _bound_weights(gt,z,lower_amp,upper_amp,cap,options):
    """Adjoints of the interval ends, with the tie test and the t-cap switch.

    Interval ends that are the amplitude bound move with the eigenvalues; ends
    at the t cap do not. An end within bound_switch_atol of the cap is a tie
    (GaugeRegularityError, first source reported, lower end before upper end).
    """
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
    return lower_weight,upper_weight,lower_on,upper_on


def _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options):
    """Eigenvalue adjoint of gt*(c'(O)+z*h'(O)), t=c+h*z, for every source at once.

    On the partition-map eigenvalues (two.map_eigenvalues); see _bound_weights
    for the cap and tie rules.
    """
    rho,cap = _partition_limits(two.amplitude_bound,amplitude_limit,max_abs_t)
    ell = np.asarray(two.map_eigenvalues,dtype=np.float64)
    r,n = ell.shape
    sources = two.sources
    lower_amp,upper_amp = interval_logs(ell,sources,two.tau2,two.partition,math.log(rho))
    lower_weight,upper_weight,lower_on,upper_on = _bound_weights(gt,z,lower_amp,upper_amp,cap,options)
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


def _leg_bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options):
    """_bound_pullback through the per-spin map: the eigenvalue adjoint (2, r, n).

    Each interval end is set by the leg spin with the largest amplitude
    (partitions.interval_logs). Spins that tie for it share the end equally:
    at equal spins (a tied shift) that gives exactly the tied derivative,
    while a per-spin derivative there averages the two one-sided ones (the
    maximum has a kink).
    """
    rho,cap = _partition_limits(two.amplitude_bound,amplitude_limit,max_abs_t)
    lam = np.asarray(two.spin_eigenvalues,dtype=np.float64)
    _,r,n = lam.shape
    sources = two.sources
    log_rho = math.log(rho)
    logs,_ = leg_log_bases(lam,sources,math.log(two.tau2),two.partition)
    top = np.max(logs,axis=2)
    lower_amp,upper_amp = top[:,1]-log_rho,log_rho-top[:,0]
    lower_weight,upper_weight,lower_on,upper_on = _bound_weights(gt,z,lower_amp,upper_amp,cap,options)
    # dL/d top[:, leg]: the upper end is log(rho) - top_L, the lower end top_R - log(rho).
    end = np.stack([-np.where(upper_on,upper_weight,0.),np.where(lower_on,lower_weight,0.)],axis=1)
    binding = (logs == top[:,:,None]) & np.isfinite(top)[:,:,None]
    share = binding/np.maximum(np.sum(binding,axis=2,keepdims=True),1)
    values,used = leg_values(lam,sources),leg_usage(sources)
    with np.errstate(divide="ignore",invalid="ignore"):
        g_leg = np.where(binding,end[:,:,None]*share/values,0.)
    if two.partition == "balanced_log":
        # top_L carries -log g_L/2 + log g_R/2, top_R the opposite.
        g_leg = g_leg+(.5*(end[:,::-1]-end))[:,:,None]*_log_scale_gradient(values,used)
    g_leg = np.where(used,g_leg,0.)
    result = np.bincount(_leg_targets(sources,r,n).reshape(-1),weights=g_leg.reshape(-1),minlength=2*r*n)
    return result.reshape(2,r,n)


def _per_spin_one_body(one):
    """The same one-body channels on per-spin (here equal) eigensystems, X (2, n, n)."""
    X = np.stack([one.X,one.X])
    eps = np.stack([one.eigenvalues,one.eigenvalues])
    X.setflags(write=False)
    eps.setflags(write=False)
    return replace(one,X=X,eigenvalues=eps)


def _one_body_pullback(one,g_one,gY,options,wall=None,checked=False):
    """dL/dh through the one-body eigensystem: (G_h, G_wall), each (k, n, n).

    k = 2 with per-spin bases, else 1 (one shared basis). The eigenvalue
    adjoint is -tau1 dL/dd for every channel d = (mu_s - eps_k) tau1, the
    eigenvector adjoint gY (loss_pullback). G_wall is the part from one-body
    shifts held on their lower bound mu_s = max(eps_s) - rho_1/tau1 (wall[s],
    relaxed mode), which moves with the largest eigenvalue.
    """
    per_spin = one.split
    eps = np.asarray(one.eigenvalues,dtype=np.float64)
    rows = eps if per_spin else eps[None]
    Y = np.asarray(one.X) if per_spin else np.asarray(one.X)[None]
    gY = np.asarray(gY) if per_spin else np.asarray(gY)[None]
    k,n = rows.shape
    table = one.table
    row = table.spin if per_spin else np.zeros_like(table.spin)
    ge = np.bincount(row*n+table.mode,weights=-one.tau1*g_one,minlength=k*n).reshape(k,n)
    gw = np.zeros((k,n))
    if wall is not None:
        for s in one.spins:
            if wall[s]:
                i = s if per_spin else 0
                gw[i,int(np.argmax(rows[i]))] += one.tau1*math.fsum(g_one[table.spin == s].tolist())
    gaps = _eigengaps(rows)
    if not checked:
        _check_one_body_gaps(eps,options,gaps)
    G = _eigh_pullback(rows,Y,ge,gY,gaps)
    return G,(Y*gw[:,None,:])@np.swapaxes(Y,1,2)


def _split_shift_gradient(g_lam,P,nelec):
    """dL/dmu[g,s] = -sum_a dL/dlambda[s,g,a] + N_s sum_s' P[g,s'] - P[g,s], shape (r, 2).

    g_lam (2, r, n): the eigenvalue adjoint per spin, moving bounds included;
    P (r, 2): <dL/dh_s, L_g>, zero for a spin without channels.
    """
    return -g_lam.sum(axis=2).T+nelec[None,:]*P.sum(axis=1)[:,None]-P


def _shift_wall(result):
    """Per-spin flags: the one-body shift was optimized and sits on its lower bound."""
    if not result.optimize_shift:
        return None
    one = result.decomposition.one_body
    lower = shift_bounds(one,result.one_body_limit)
    return tuple(lower[s] is not None and one.shift[s] <= lower[s] for s in (0,1))


def _assemble(problem,decomposition,options,mode,*,bounds=None,amplitude_limit=None,
              max_abs_t=30.,inner_result=None,loss_kind="weighted",check=True,regular=False,
              space=None,cholesky_gradient=None,wall=None):
    """Loss and gauge derivative; check=False trusts a GaugeEvaluator decomposition.

    regular=True means check_gauge_regularity already passed at this gauge
    (GaugeEvaluator runs it before any inner work, with the one-body eigengaps
    when they matter); otherwise the zero-eigenvalue and eigengap tests run
    here, in the order the per-factor code used.

    cholesky_gradient: None, 'tied' or 'split' (_cholesky_gradient_kind).
    Without a Cholesky shift, None is exactly the unshifted derivative; every
    other case goes through _assemble_shifted. wall: per-spin flags of one-body
    shifts optimized onto their lower bound (_shift_wall, relaxed mode).
    """
    if check:
        _validate_state(problem,decomposition)
    two = decomposition.two_body
    kind = _cholesky_gradient_kind(cholesky_gradient,two)
    if kind is not None:
        if check and two.cholesky_shift is None:
            _check_one_body_system(problem,decomposition)
        return _assemble_shifted(problem,decomposition,options,mode,kind,bounds=bounds,
                                 amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                                 inner_result=inner_result,loss_kind=loss_kind,check=check,
                                 regular=regular,space=space,wall=wall)
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


def _assemble_shifted(problem,decomposition,options,mode,kind,*,bounds,amplitude_limit,max_abs_t,
                      inner_result,loss_kind,check,regular,space,wall):
    """_assemble with h_sigma = h + sum_g (m_g - mu[g,sigma]) L_g(O) and the shift gradient.

    Two-body: per-spin leg pullbacks for a split decomposition or a 'split'
    request, else today's pullbacks on ell - mu. One-body: the eigh pullback
    of h_sigma (per spin when needed; equal spins then share an eigensystem).
    With m = mu @ N and P[g, s] = <dL/dh_s, L_g>,

        dL/dL_g += sum_s (m_g - mu[g,s]) dL/dh_s,
        dL/dmu[g,s] = -sum_a dL/dlambda[s,g,a] + N_s sum_s' P[g,s'] - P[g,s].

    The tied gradient is the sum over spins (column 0 for aa_only, whose
    mu_down is 0); on today's layout it is -sum_a dL/dlambda + (N-1) sum_s P.
    Moving bounds (interval ends, one-body shifts on their wall) go to
    bound_correction_O.
    """
    two,one = decomposition.two_body,decomposition.one_body
    if two.nelec is None:
        raise ValueError("A Cholesky-shift gradient requires particle numbers: pass nelec to prepare_problem")
    mu = two.cholesky_shift
    legs = two.split or kind == "split"
    work = decomposition
    if legs and not one.split and len(one.spins) == 2:
        # Per-spin h adjoints need per-spin bases.
        one = _per_spin_one_body(one)
        work = replace(decomposition,one_body=one)
        if space is not None and not space.split:
            space = None
    loss,gamp,gX,g_one,gY = loss_pullback(work,loss_kind,check,space,with_one_body=True)
    if legs:
        g_lam,gt = _leg_amplitude_pullback(two,gamp,options,checked=regular)
        extra_lam = np.zeros_like(g_lam)
        if bounds is not None:
            z = _normalized_state(two,bounds,options)
            extra_lam = _leg_bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options)
        gell,extra = g_lam.sum(axis=0),extra_lam.sum(axis=0)
    else:
        gell,gt = _amplitude_pullback(two,gamp,options,checked=regular)
        extra = np.zeros_like(gell)
        if bounds is not None:
            z = _normalized_state(two,bounds,options)
            extra = _bound_pullback(two,gt,z,amplitude_limit,max_abs_t,options)
    ell,X = two.gauge.eigenvalues,two.gauge.X
    gaps = _eigengaps(ell)
    if not regular:
        _check_eigengaps(ell,options,gaps)
    G_h,G_wall = _one_body_pullback(one,g_one,gY,options,wall,checked=regular)
    gL = _eigh_pullback(ell,X,gell,gX,gaps)
    extraL = (X*extra[:,None,:])@np.swapaxes(X,1,2)
    chol = np.asarray(two.gauge.chol,dtype=np.float64)
    nelec = np.asarray(two.nelec,dtype=np.float64)
    # Rows of G_h: one per spin, or one shared basis (the matrix of the first spin with channels).
    rows = [0,1] if one.split else [one.spins[0]]
    if mu is not None:
        mu = np.asarray(mu,dtype=np.float64)
        coefficient = (mu@nelec)[:,None]-mu                      # dh_s/dL_g
        gL = gL+np.einsum("gk,kpq->gpq",coefficient[:,rows],G_h)
        extraL = extraL+np.einsum("gk,kpq->gpq",coefficient[:,rows],G_wall)
    P = np.einsum("kpq,gpq->gk",G_h+G_wall,chol)
    if legs:
        full = np.zeros((len(chol),2))
        full[:,rows] = P
        split = _split_shift_gradient(g_lam+extra_lam,full,nelec)
        if kind == "split":
            gmu = split
        else:
            gmu = split[:,0] if two.policy == "aa_only" else split.sum(axis=1)
    else:
        gmu = -(gell+extra).sum(axis=1)+(nelec.sum()-1)*P.sum(axis=1)
    original = problem.data.chol.reshape(problem.data.nchol,-1)
    fixed = gL.reshape(problem.data.nchol,-1)@original.T
    correction = extraL.reshape(problem.data.nchol,-1)@original.T
    ambient = fixed+correction
    if not all(np.isfinite(x).all() for x in (gL,gell,gt,ambient,gmu)):
        raise FloatingPointError("Nonfinite gauge gradient")
    local = ambient@two.gauge.O.T
    generator = (local-local.T)/2
    tangent = generator@two.gauge.O
    return GaugeGradient(
        float(loss),decomposition,mode,_readonly(ambient),_readonly(tangent),
        _readonly(generator),_readonly(fixed),_readonly(correction),_readonly(gt),
        _readonly(gell+extra),inner_result=inner_result,cholesky_gradient=_readonly(gmu))


def fixed_partition_gradient(problem,decomposition,*,options=None,loss_kind="weighted",
                             cholesky_gradient=None):
    """d L/dO with physical log ratios t (and the shift) held fixed, including eigh.

    loss_kind selects the unnormalized L_w ('weighted', default) or L_raw. No partition-bound
    response is part of this derivative. Trial gauges with the same t must
    independently satisfy the open amplitude domain. Without a Cholesky shift
    the one-body channels do not depend on O; with one, h_sigma does (through
    the rotated factors), and gradient_O includes that term.

    cholesky_gradient: None (default) reports dL/d of the decomposition's own
    Cholesky shift in GaugeGradient.cholesky_gradient, (r,) when tied and
    (r, 2) when split, and nothing without a shift. 'tied' or 'split' choose
    the form, also at zero shift; they need nelec and simple h_sigma eigenvalues.
    """
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_t",
                     loss_kind=loss_kind_name(loss_kind),cholesky_gradient=cholesky_gradient)


def _normalized_gradient(problem,decomposition,options,amplitude_limit,max_abs_t,loss_kind,
                         cholesky_gradient,wall=None):
    two = decomposition.two_body
    bounds = _partition_intervals(two.map_eigenvalues,two.sources,two.tau2,
                                 two.partition,two.amplitude_bound,
                                 amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)
    return _assemble(problem,decomposition,_gradient_options(options),"fixed_z",
                     bounds=bounds,amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                     loss_kind=loss_kind_name(loss_kind),cholesky_gradient=cholesky_gradient,wall=wall)


def normalized_partition_gradient(problem,decomposition,*,amplitude_limit=None,
                                  max_abs_t=30.,options=None,loss_kind="weighted",
                                  cholesky_gradient=None):
    """Gauge derivative at fixed z in t=(lower+upper)/2+(upper-lower)*z/2.

    This supports both maps and capped/asymmetric intervals. It is different
    from a fixed-t derivative even when no partition optimizer is involved.
    The one-body shift is held fixed. Its bound depends on h_sigma only, so
    on O only through a Cholesky shift; a fixed shift has no bound term.
    cholesky_gradient as in fixed_partition_gradient (at the same fixed z).
    """
    return _normalized_gradient(problem,decomposition,options,amplitude_limit,max_abs_t,
                                loss_kind,cholesky_gradient)


def _relaxed_gradient(problem,result,options,regular=False,space=None,cholesky_gradient=None):
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
                               regular=regular,space=space,cholesky_gradient=cholesky_gradient,
                               wall=_shift_wall(result))
    except (GaugeRegularityError,FloatingPointError,np.linalg.LinAlgError) as exc:
        # A rejected outer trial can still account for the completed inner work.
        exc.inner_result = result
        raise
    return replace(derivative,mode="relaxed",inner_result=result)


def relaxed_gauge_gradient(problem,result,*,options=None,cholesky_gradient=None):
    """Envelope gradient at a converged step-4 solution, including moving bounds.

    Uses the recorded amplitude_limit/max_abs_t, not guessed default bounds.
    Assumes a regular consistently tracked local inner minimum; no claim of
    a global inner solution is made for a general multi-source problem.

    The Cholesky shift is an outer variable here (held fixed by the inner
    solve), so its gradient is an ordinary partial derivative at the envelope.
    A relaxed one-body shift on its lower bound moves with max(eps_sigma)
    when h_sigma depends on O or on the Cholesky shift; that term is included.
    cholesky_gradient as in fixed_partition_gradient.
    """
    if not isinstance(result,PartitionOptimizationResult):
        raise TypeError("result must be PartitionOptimizationResult")
    if not result.converged:
        raise InnerConvergenceError("Relaxed gradient requires a converged partition solve",result)
    try:
        derivative = _normalized_gradient(
            problem,result.decomposition,options,result.amplitude_limit,result.max_abs_t,
            result.loss_kind,cholesky_gradient,_shift_wall(result))
    except (GaugeRegularityError,FloatingPointError,np.linalg.LinAlgError) as exc:
        # A rejected outer trial can still account for the completed inner work.
        exc.inner_result = result
        raise
    two = result.decomposition.two_body
    bounds = _partition_intervals(two.map_eigenvalues,two.sources,two.tau2,
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

    cholesky_shift: None, (r,) tied or (r, 2) per-spin shifts of the rotated
    factors (channels.cholesky_shift_array), the default for every method;
    each method takes a per-call cholesky_shift= (zeros for none). With a
    shift the one-body matrices h_sigma depend on the gauge: they are
    diagonalized per trial, checked for eigengaps with the factor checks, and
    gradient_O includes their gauge dependence.

    cholesky_gradient: None (default) reports GaugeGradient.cholesky_gradient
    for the shift in use, in its own form (none without a shift); 'tied' or
    'split' always report it in that form, also at zero shift (then bare h
    must have simple eigenvalues). 'split' keeps per-spin one-body bases,
    equal at a tied shift, so the inner solve and the gradient share one mode
    space.
    """

    def __init__(self,problem,*,tau1,tau2,partition="balanced_log",amplitude_bound=1.,
                 options=None,backend="numpy",loss_kind="weighted",shift=None,cholesky_shift=None,
                 cholesky_gradient=None):
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
        self._cholesky_shift = self._shift_array(cholesky_shift)
        self._cholesky_gradient = _cholesky_gradient_option(cholesky_gradient)
        if self._cholesky_gradient is not None and problem.nelec is None:
            raise ValueError("A Cholesky-shift gradient requires particle numbers: pass nelec to prepare_problem")
        self._per_spin = self._cholesky_gradient == "split" and len(self.one_body.spins) == 2
        self._bare = None

    @property
    def backend(self):
        """Derivative backend chosen at construction (read-only)."""
        return self._backend

    @property
    def shift(self):
        """Fixed (mu_up, mu_down) used unless a call supplies or relaxes one."""
        return self.one_body.shift

    @property
    def cholesky_shift(self):
        """The engine's (r, 2) Cholesky shift, the default of every method, or None."""
        return self._cholesky_shift

    @property
    def cholesky_gradient(self):
        """The requested Cholesky-shift gradient form: None (the shift's own), 'tied' or 'split'."""
        return self._cholesky_gradient

    def with_cholesky_shift(self,cholesky_shift):
        """A copy of this evaluator whose default Cholesky shift is cholesky_shift (None or zeros: none).

        The copy shares the problem, the sources and the bare one-body
        channels; this evaluator is not changed.
        """
        other = copy.copy(self)
        other._cholesky_shift = self._shift_array(cholesky_shift)
        return other

    def with_cholesky_gradient(self,cholesky_gradient):
        """A copy of this evaluator that reports the shift gradient in the form cholesky_gradient.

        None, 'tied' or 'split', as for the constructor; this evaluator is not changed.
        """
        kind = _cholesky_gradient_option(cholesky_gradient)
        if kind is not None and self.problem.nelec is None:
            raise ValueError("A Cholesky-shift gradient requires particle numbers: pass nelec to prepare_problem")
        other = copy.copy(self)
        other._cholesky_gradient = kind
        other._per_spin = kind == "split" and len(self.one_body.spins) == 2
        other._bare = None
        return other

    def _shift_array(self,cholesky_shift):
        mu = cholesky_shift_array(cholesky_shift,self.problem.data.nchol,self.problem.policy)
        check_cholesky_shift(mu,self.problem.policy,self.problem.nelec)
        return mu

    def _mu(self,cholesky_shift):
        """The engine shift for None, else the per-call shift (zeros give None)."""
        return self._cholesky_shift if cholesky_shift is None else self._shift_array(cholesky_shift)

    def _needs_h(self,mu):
        """True when the derivative goes through the one-body eigensystem."""
        return mu is not None or self._cholesky_gradient is not None

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

    def _system(self,gauge,mu):
        """(eigenvalues, X) of the one-body matrices at this gauge and Cholesky shift.

        Per spin when the shift is split or a 'split' gradient was requested.
        Without a shift it is the cached bare-h eigensystem.
        """
        if mu is not None:
            return one_body_eigensystem(self.problem.data.h1,gauge.chol,mu,self.problem.nelec,
                                        self.one_body.spins,self._per_spin)
        if not self._per_spin:
            return self.one_body.eigenvalues,self.one_body.X
        if self._bare is None:
            pair = (np.stack([self.one_body.eigenvalues]*2),np.stack([self.one_body.X]*2))
            for array in pair:
                array.setflags(write=False)
            self._bare = pair
        return self._bare

    def _trial(self,O,mu):
        """_gauge plus the regularity checks: a singular trial fails before any inner work.

        Returns (gauge, one-body system); the system is None when the
        derivative does not go through h (no shift and no shift gradient).
        """
        gauge = self._gauge(O)
        check_gauge_regularity(gauge,self.sources,self.partition,self.options,
                               eigenvalues=map_eigenvalues(gauge.eigenvalues,mu))
        if not self._needs_h(mu):
            return gauge,None
        system = self._system(gauge,mu)
        _check_one_body_gaps(system[0],self.options)
        return gauge,system

    def _decomposition(self,gauge,t,shift=None,mu=None,system=None):
        if system is None:
            t,d = fixed_partition_amplitudes(gauge.eigenvalues,self.sources,self.tau2,t,
                                             self.partition,self.amplitude_bound)
            two = _two_body_from_amplitudes(gauge,self.problem.policy,self.problem.nelec,self.tau2,
                                            self.partition,self.amplitude_bound,self.sources,t,d)
            return FixedDecomposition(self._one_body(shift),two)
        # Shifted factors ell - mu; the one-body matrices h_sigma depend on this gauge.
        t,d = fixed_partition_amplitudes(map_eigenvalues(gauge.eigenvalues,mu),self.sources,self.tau2,t,
                                         self.partition,self.amplitude_bound)
        two = _two_body_from_amplitudes(gauge,self.problem.policy,self.problem.nelec,self.tau2,
                                        self.partition,self.amplitude_bound,self.sources,t,d,mu)
        ell,X = system
        one = _one_body_from_eigensystem(X,ell,self.one_body.tau1,
                                         self.one_body.shift if shift is None else shift_pair(shift),
                                         self.one_body.spins,self.problem.nelec,mu)
        return FixedDecomposition(one,two)

    def decomposition(self,O=None,*,t=0.,shift=None,cholesky_shift=None):
        """Fast channel construction for a trial orthogonal gauge, with fixed t and shift.

        cholesky_shift=None uses the engine's Cholesky shift; pass an explicit
        (r,) or (r, 2) array (zeros for none) to override it for this call.
        The one-body representation is the one the gradient methods use.
        """
        mu = self._mu(cholesky_shift)
        gauge = self._gauge(O)
        return self._decomposition(gauge,t,shift,mu,self._system(gauge,mu) if self._needs_h(mu) else None)

    def fixed(self,O=None,*,t=0.,shift=None,cholesky_shift=None):
        """Loss and derivative with t (and both shifts) fixed."""
        mu = self._mu(cholesky_shift)
        gauge,system = self._trial(O,mu)
        return _assemble(self.problem,self._decomposition(gauge,t,shift,mu,system),self.options,
                         "fixed_t",loss_kind=self.loss_kind,check=False,regular=True,
                         cholesky_gradient=self._cholesky_gradient)

    def _bounds(self,gauge,mu,amplitude_limit,max_abs_t):
        return _partition_intervals(map_eigenvalues(gauge.eigenvalues,mu),self.sources,self.tau2,
                                    self.partition,self.amplitude_bound,
                                    amplitude_limit=amplitude_limit,max_abs_t=max_abs_t)

    def fixed_z(self,O=None,*,z=0.,amplitude_limit=None,max_abs_t=30.,shift=None,cholesky_shift=None):
        """Evaluate a gauge at fixed normalized coordinates z in [-1,1] and fixed shifts."""
        mu = self._mu(cholesky_shift)
        gauge,system = self._trial(O,mu)
        bounds = self._bounds(gauge,mu,amplitude_limit,max_abs_t)
        z = _coordinates(z,len(self.sources),"z")
        if np.any(np.abs(z)>1):
            raise ValueError("z must lie in [-1,1]")
        t = bounds[:,0]+(z+1)*(bounds[:,1]-bounds[:,0])/2
        t = np.where(z == 1,bounds[:,1],np.where(z == -1,bounds[:,0],t))
        return _assemble(self.problem,self._decomposition(gauge,t,shift,mu,system),self.options,"fixed_z",
                         bounds=bounds,amplitude_limit=amplitude_limit,max_abs_t=max_abs_t,
                         loss_kind=self.loss_kind,check=False,regular=True,
                         cholesky_gradient=self._cholesky_gradient)

    def _shift_lower(self,system,one_body_limit):
        """shift_bounds of the one-body system at this trial (before its channels exist)."""
        if system is None:
            return shift_bounds(self.one_body,one_body_limit)
        rho = one_body_limit_value(one_body_limit)
        eps = system[0]
        spins = self.one_body.spins
        return tuple(float(np.max(eps[s] if np.ndim(eps) == 2 else eps)-rho/self.one_body.tau1)
                     if s in spins else None for s in (0,1))

    def relaxed(self,O=None,*,t0=None,options=None,cholesky_shift=None):
        """Relax partitions (and optionally the shift) at THIS gauge; envelope gradient.

        options: InnerOptions (defaults when None), the inner-solver settings.
        Warm-start log ratios are projected into the current intervals. With
        t0=None, use interval midpoints. With options.optimize_shift=True the
        shift starts at mu*(t0) (no shift seed) and is relaxed together with t;
        the engine shift is used only if it is feasible, otherwise the lower
        bound. Every call, including line-search or finite-difference trial
        gauges, performs its own converged inner solve. The Cholesky shift is
        held fixed (an outer variable).
        """
        options = _inner_options(options)
        mu = self._mu(cholesky_shift)
        gauge,system = self._trial(O,mu)
        bounds = self._bounds(gauge,mu,options.amplitude_limit,options.max_abs_t)
        t = bounds.mean(axis=1) if t0 is None else np.clip(
            _coordinates(t0,len(self.sources),"t0"),bounds[:,0],bounds[:,1])
        shift = None
        if options.optimize_shift:
            lower = self._shift_lower(system,options.one_body_limit)
            shift = tuple(m if low is None else max(m,low)
                          for m,low in zip(self.one_body.shift,lower))
        # One mode space per trial gauge: the inner solve and the envelope gradient share it.
        space = ModeSpace(gauge.X,self.one_body.X if system is None else system[1])
        inner = _solve_partitions(self._decomposition(gauge,t,shift,mu,system),options,self.loss_kind,space)
        return _relaxed_gradient(self.problem,inner,self.options,regular=True,space=space,
                                 cholesky_gradient=self._cholesky_gradient)


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
