"""Regenerate systems.npz: molecular integrals for the golden and benchmark cases.

Requires PySCF; the tests and the benchmark only read systems.npz. Run from the
repository root:

    python ipie/hamiltonians/optimize_cholesky/tests/reference/make_systems.py

Each system is RHF-MO integrals with a tight modified Cholesky (tol 1e-10).
The stored eri is rebuilt from the stored chol, so prepare_problem's exact
ERI-reconstruction check passes. nelec selects the sector policy:
(1,1) -> ab_only, (N,0) -> aa_only, otherwise hybrid.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MOLECULES = {
    # name: (geometry in Angstrom, basis, nelec, description)
    "h2_631g": ("H 0 0 0; H 0 0 0.74", "6-31g", (1, 1), "H2 6-31G, singlet sector (ab_only)"),
    "h2_631g_triplet": ("H 0 0 0; H 0 0 0.74", "6-31g", (2, 0), "H2 6-31G, (2,0) sector (aa_only)"),
    "h4_sto3g": ("H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", "sto-3g", (2, 2),
                 "linear H4 1.0 A, STO-3G (hybrid)"),
    "lih_sto3g": ("Li 0 0 0; H 0 0 1.6", "sto-3g", (2, 2), "LiH 1.6 A, STO-3G (hybrid)"),
    "h4_631g": ("H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", "6-31g", (2, 2),
                "linear H4 1.0 A, 6-31G (hybrid, benchmark size)"),
    "h2_ccpvdz": ("H 0 0 0; H 0 0 0.74", "cc-pvdz", (1, 1), "H2 cc-pVDZ (ab_only, target size)"),
}


def build(geometry, basis, tol=1e-10):
    from pyscf import ao2mo, gto, scf
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."]*5)))
    from ipie.utils.linalg import modified_cholesky
    mol = gto.M(atom=geometry, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    C = mf.mo_coeff
    n = C.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), n)                  # eri[p,q,r,s] = (pq|rs)
    chol = modified_cholesky(eri.reshape(n*n, n*n), tol=tol, verbose=False, cmax=n).reshape(-1, n, n)
    chol = 0.5*(chol + chol.transpose(0, 2, 1))
    eri_chol = np.einsum("gpr,gqs->prqs", chol, chol)
    return dict(h1=h1, eri=eri_chol, chol=chol, ecore=mol.energy_nuc(), ehf=mf.e_tot,
                cholesky_residual=np.abs(eri-eri_chol).max())


def main(path=os.path.join(HERE, "systems.npz")):
    arrays = {}
    for name, (geometry, basis, nelec, description) in MOLECULES.items():
        data = build(geometry, basis)
        for key, value in data.items():
            arrays[f"{name}/{key}"] = np.asarray(value)
        arrays[f"{name}/nelec"] = np.array(nelec, dtype=np.int64)
        arrays[f"{name}/description"] = np.array(description)
        print(f"{name:16s} n={data['h1'].shape[0]:2d} nchol={data['chol'].shape[0]:3d} "
              f"nelec={nelec} max|eri-LL|={data['cholesky_residual']:.1e}  {description}")
    np.savez_compressed(path, **arrays)
    print("wrote", path)


if __name__ == "__main__":
    main()
