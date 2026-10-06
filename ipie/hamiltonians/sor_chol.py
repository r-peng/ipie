import numpy as np
import pickle

def complete_orthogonal_basis(c, tol=1e-12):
    c = np.asarray(c, dtype=float)
    c = c / np.linalg.norm(c)
    rows = [c]
    for e in np.eye(c.size):
        v = e.copy()
        Q = np.asarray(rows)
        v -= Q.T @ (Q @ v)
        nrm = np.linalg.norm(v)
        if nrm > tol:
            rows.append(v / nrm)
        if len(rows) == c.size:
            break
    return np.asarray(rows)

def rotate_2x2_cholesky_make_identity(chol, tol=1e-12):
    chol = np.asarray(chol)
    assert chol.ndim == 3
    assert chol.shape[1:] == (2, 2)
    nchol = chol.shape[0]
    # Traceless symmetric part:
    # [[ z, x],
    #  [ x,-z]]
    T = np.empty((nchol, 2))
    T[:, 0] = 0.5 * (chol[:, 0, 0] - chol[:, 1, 1])
    T[:, 1] = chol[:, 0, 1]

    # Need c @ T = 0.
    # Null space of T.T obtained from SVD.
    _, s, Vh = np.linalg.svd(T.T, full_matrices=True)
    rank = np.sum(s > tol * max(s[0], 1.0))
    if rank >= nchol:
        raise ValueError("No Cholesky-space combination proportional to identity.")
    c = Vh[rank].copy()
    c /= np.linalg.norm(c)

    # Complete c to an orthogonal Cholesky-space transformation.
    O = complete_orthogonal_basis(c)

    chol_rot = np.einsum('ag,gij->aij',O,chol)
    L0 = chol_rot[0]
    alpha = 0.5 * np.trace(L0)
    residual = np.linalg.norm(L0 - alpha * np.eye(2),ord='fro')
    return chol_rot, O, c, residual

def normalized_commutator(A, B, eps=1e-16):
    nA = np.linalg.norm(A, ord='fro')
    nB = np.linalg.norm(B, ord='fro')
    denom = max(nA * nB, eps)
    comm = A @ B - B @ A
    return np.linalg.norm(comm, ord='fro') / denom

def commutator_matrix(chol):
    nchol = chol.shape[0]
    norms = np.linalg.norm(chol.reshape(nchol, -1),axis=1)
    C = np.zeros((nchol, nchol))
    for i in range(nchol):
        for j in range(i + 1, nchol):
            d = normalized_commutator(chol[i], chol[j])
            C[i, j] = d
            C[j, i] = d
    return C

def group_commuting_cholesky(chol, tol=1e-10):
    chol = np.asarray(chol)
    nchol = chol.shape[0]
    C = commutator_matrix(chol)

    # Number of matrices with which each matrix does NOT commute.
    # Put the most difficult matrices first.
    conflict_degree = np.sum(C > tol, axis=1)
    order = np.argsort(-conflict_degree)
    groups = []
    for i in order:
        placed = False
        for group in groups:
            # Complete-linkage condition:
            # i must commute with EVERY member already in the group.
            if all(C[i, j] <= tol for j in group):
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    # Optional: sort indices inside each group
    groups = [sorted(g) for g in groups]
    return groups, C

def common_eigenbasis(chol,group,seed=7,ntry=10):
    rng = np.random.default_rng(seed)
    mats = np.asarray(chol)[group]
    ng, nb, _ = mats.shape

    best = None
    denom = sum([np.linalg.norm(L, ord='fro')**2 for L in mats])
    for _ in range(ntry):
        coeff = rng.normal(size=ng)
        A = np.einsum('g,gij->ij', coeff, mats)
        # Make numerical symmetry explicit
        A = 0.5 * (A + A.T)
        _, U = np.linalg.eigh(A)
        rotated = np.einsum('pi,gpq->giq',U,mats)
        rotated = np.einsum('giq,qj->gij',rotated,U)
        #rotated = np.einsum('pi,gpq,qj->gij',U,mats,U,optimize=True)

        diag = np.einsum('gii->gi', rotated)
        off = rotated.copy()
        idx = np.arange(nb)
        off[:, idx, idx] = 0.0

        residual = (np.sum(np.abs(off)**2) / max(denom, 1e-30))**0.5
        if best is None or residual < best[0]:
            best = (residual, U, diag)

    residual, U, eps = best
    return U, eps, residual

def build_commuting_groups(chol,comm_tol=1e-10,basis_tol=1e-10,seed=7):
    groups, C = group_commuting_cholesky(chol,tol=comm_tol)
    result = []
    for ig, group in enumerate(groups):
        if len(group) == 1:
            gamma = group[0]
            e, U = np.linalg.eigh(chol[gamma])
            eps = e[None, :]
            residual = 0.0
        else:
            U, eps, residual = common_eigenbasis(chol,group,seed=seed + ig)

        if residual > basis_tol:
            print(
                f"WARNING: group {ig} has joint-diagonalization "
                f"residual {residual:.3e}"
            )

        W = eps.T @ eps
        print('group=',group)
        print('eps=',eps)
        print('res=',residual)
        print('X:')
        print(U)
        print('K:')
        print(W)

        result.append({
            'indices': group,
            'X': U,
            'eps': eps,
            'W': W,
            'residual': residual,
            'isometry':True,
        })
    return result, C

