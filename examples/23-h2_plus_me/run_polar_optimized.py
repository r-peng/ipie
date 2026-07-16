import numpy as np
import h5py, time
from ipie.hamiltonians.sor_base import QCSOR
from ipie.utils.linalg import modified_cholesky

np.set_printoptions(suppress=True, precision=6)

N = 10
Nr, Ntheta, Nphi = 11, N * 4, N * 2 + 1
dr, dtheta, dphi = 1.0 / (Nr - 1), 2 * np.pi / Ntheta, np.pi / (Nphi - 1)
rs = np.arange(Nr) * dr
thetas = np.arange(Ntheta) * dtheta
phis = np.arange(Nphi) * dphi

print(rs)
print(thetas)
print(phis)

print("computing all mos...")
t0 = time.time()
ct = np.cos(thetas)
st = np.sin(thetas)
sp = np.sin(phis)
cp = np.cos(phis)

mos_grid = np.empty((Nr, Ntheta, Nphi, 4), dtype=np.float64)
mos_grid[..., 0] = rs[:, None, None] * ct[None, :, None] * sp[None, None, :]
mos_grid[..., 1] = rs[:, None, None] * st[None, :, None] * sp[None, None, :]
mos_grid[..., 2] = rs[:, None, None] * cp[None, None, :]
mos_grid[..., 3] = np.sqrt(np.maximum(0.0, 1.0 - rs**2))[:, None, None]
Nh = Nr * Ntheta * Nphi
mos = mos_grid.reshape(Nh, 4)
print("computing mo time=", time.time() - t0)

R = 1.1
xc = "b3lyp"
print(f"############### R={R:.1f} #################")
with h5py.File(f"integrals/R{R:.1f}_{xc}.h5", "r") as f:
    h1e = f["hcore"][:]
    eri = f["eri"][:]
    mo_coeff = f["mo_coeff"][:]

nsite = eri.shape[0]
cmax = nsite**2
M = eri.reshape((nsite**2, nsite**2))
print("eri symmetry=", np.linalg.norm(M - M.T))
chol = modified_cholesky(M, cmax=cmax).reshape(-1, nsite, nsite)

ham = QCSOR(apply_spin_down=False)
iprint = 1
dt = 0.01
ham.decompose_h1(h1e, 1.0 / dt, iprint=iprint)
ham.decompose_h2(chol, 5.0, iprint=iprint)
ham.parse_decomposition()


def flatten_idx(ir, itheta, iphi):
    return (ir * Ntheta + itheta) * Nphi + iphi


def mo2ix(mo, thresh=1e-8):
    """Scalar version, kept for one-off use and checking."""
    mo = np.array(mo, dtype=np.float64, copy=True)
    norm = np.linalg.norm(mo)
    mo /= norm
    if mo[3] < -thresh:
        norm *= -1.0
        mo *= -1.0

    r = np.sqrt(max(0.0, 1.0 - mo[3] ** 2))
    if r < thresh:
        return 0, norm

    ir = int(np.rint(r / dr) + 1e-6)
    mo /= r

    if mo[2] < -1 + thresh:
        return flatten_idx(ir, 0, Nphi - 1), norm
    if mo[2] > 1 - thresh:
        return flatten_idx(ir, 0, 0), norm

    iphi = int(np.rint(np.arccos(np.clip(mo[2], -1.0, 1.0)) / dphi) + 1e-6)
    theta = np.mod(np.arctan2(mo[1], mo[0]), 2 * np.pi)
    itheta = int(np.rint(theta / dtheta) + 1e-6) % Ntheta
    return flatten_idx(ir, itheta, iphi), norm


def mo2ix_vec(vecs, thresh=1e-8):
    """Vectorized version of mo2ix for shape (..., 4)."""
    orig_shape = vecs.shape[:-1]
    v = vecs.reshape(-1, 4).astype(np.float64, copy=True)

    norms = np.linalg.norm(v, axis=1)
    if np.any(norms < thresh):
        raise ValueError("Encountered a near-zero MO vector; cannot map to grid.")

    u = v / norms[:, None]
    flip = u[:, 3] < -thresh
    fac = norms.copy()
    fac[flip] *= -1.0
    u[flip] *= -1.0

    r = np.sqrt(np.maximum(0.0, 1.0 - u[:, 3] ** 2))
    out_ix = np.zeros(u.shape[0], dtype=np.int64)

    active = r >= thresh
    if np.any(active):
        idx = np.nonzero(active)[0]
        rr = r[idx]
        xyz = u[idx, :3] / rr[:, None]

        ir = np.rint(rr / dr).astype(np.int64)
        ir = np.clip(ir, 0, Nr - 1)

        z = np.clip(xyz[:, 2], -1.0, 1.0)
        iphi = np.rint(np.arccos(z) / dphi).astype(np.int64)
        iphi = np.clip(iphi, 0, Nphi - 1)

        theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)
        itheta = np.rint(theta / dtheta).astype(np.int64) % Ntheta

        # At the polar caps theta is irrelevant; match the scalar routine.
        itheta[z < -1.0 + thresh] = 0
        itheta[z > 1.0 - thresh] = 0

        out_ix[idx] = flatten_idx(ir, itheta, iphi)

    return out_ix.reshape(orig_shape), fac.reshape(orig_shape)


print("precomputing Umos...")
t0 = time.time()
trial = mo_coeff[0, :, 0]
Us = np.asarray([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
Umos = np.einsum("axy,iy->aix", Us, mos, optimize=True)
print("computing Umo time=", time.time() - t0)

ix_path = f"integrals/R{R:.1f}_{xc}_Umos_ixs_{Nr}_{N}.npy"
fac_path = f"integrals/R{R:.1f}_{xc}_Umos_fac_{Nr}_{N}.npy"
try:
    Umos_ixs = np.load(ix_path)
    Umos_fac = np.load(fac_path)
except FileNotFoundError:
    print("precomputing index and fac maps...")
    t0 = time.time()
    Umos_ixs, Umos_fac = mo2ix_vec(Umos)
    np.save(ix_path, Umos_ixs)
    np.save(fac_path, Umos_fac)
    print("computing idx map time=", time.time() - t0)

# Precompute the weighted map. This makes each matvec a single bincount.
map_ix = Umos_ixs.ravel()
map_fac = (np.asarray(ham.a)[:, None] * Umos_fac).ravel()

trial_mos = mos @ trial


def mat_vec(psi):
    # Equivalent to:
    # psi_new[Umos_ixs[ix1, ix2]] += ham.a[ix1] * psi[ix2] * Umos_fac[ix1, ix2]
    vals = map_fac * np.broadcast_to(psi, Umos_fac.shape).ravel()
    return np.bincount(map_ix, weights=vals, minlength=Nh)


def mat_vec_chunked(psi, chunk_terms=512):
    """Lower-memory variant if map_fac is too large for your production run."""
    out = np.zeros(Nh, dtype=np.result_type(psi, Umos_fac))
    for start in range(0, ham.nterms, chunk_terms):
        stop = min(start + chunk_terms, ham.nterms)
        vals = (np.asarray(ham.a)[start:stop, None] * Umos_fac[start:stop]) * psi[None, :]
        out += np.bincount(Umos_ixs[start:stop].ravel(), weights=vals.ravel(), minlength=Nh)
    return out


check = False
if check:
    psi_test = np.random.rand(Nh) * 2 - 1
    G1 = mat_vec(psi_test)
    G2 = mat_vec_chunked(psi_test, chunk_terms=32)
    print("check matvec=", np.linalg.norm(G1 - G2), np.linalg.norm(G1))
    raise SystemExit

psi = np.zeros(Nh)
ix, _ = mo2ix(trial)
psi[ix] = 1.0

Nstep = 200
E = []
for i in range(Nstep):
    Gpsi = mat_vec(psi)
    denom = np.dot(trial_mos, psi)
    Ei = np.dot(trial_mos, Gpsi) / denom
    Ei = ham.Lambda[-1] * (1.0 - Ei)
    E.append(Ei)
    print(i, Ei)
    psi = Gpsi
