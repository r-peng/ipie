"""Analytic float64 pullbacks for overlap norms, symmetric eigh and gauges."""

import numpy as np
from scipy.linalg import expm_frechet

from .models import GaugeRegularityError, GradientOptions
from .objective import commutator_norm2
from .partitions import _readonly
from .prepare import exponential_gauge, givens_gauge, skew_from_parameters
from .validation import _real_array

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


