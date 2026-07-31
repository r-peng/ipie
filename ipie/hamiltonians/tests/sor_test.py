import numpy as np
import itertools,scipy
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        get_all_configs_u1,
        string_act,
        count_double_occupancy,
        apply_a_dag_dense_sign,
        get_config_from_occ,
)
np.set_printoptions(suppress=True,precision=6,linewidth=100000)

def get_MB_kappa(ham,ix,basis,basis_map):
    chol_ix,spin,ps,ds = ham.get_term(ix)
    v = ham.chol_basis[chol_ix]
    g = np.log(1.+ds)
    kappa = [None] * 2
    if spin==(0,1):
        for s,p in enumerate(ps):
            ks = np.outer(v[:,p],v[:,p]*g[s])
            kappa[s] = quadratic2MB(ks,basis,basis_map,s) 
    else:
        s = spin[0]
        ks = np.einsum('xr,yr,r->xy',v[:,ps],v[:,ps],g)
        kappa[s] = quadratic2MB(ks,basis,basis_map,s) 
    return kappa

def get_MB_gf(ham,basis,basis_map):
    H = 0
    for ix,ai in enumerate(ham.a): 
        kappa = get_MB_kappa(ham,ix,basis,basis_map)
        U = None
        for spin,k in enumerate(kappa):
            if k is None:
                continue
            Us = scipy.linalg.expm(k)
            if U is None:
                U = Us
            else:
                U = np.dot(U,Us)
        H += ai*U
    return H

def quadratic2MB(M,basis,basis_map,spin,thresh=1e-6):
    if len(M.shape)==1:
        M = np.diag(M)
    H = np.zeros((len(basis),)*2)
    for (p,q) in itertools.product(range(M.shape[0]),repeat=2):
        if np.absolute(M[p,q])<thresh:
            continue
        for ix1,cf1 in enumerate(basis):
            ops = (2*p+spin,'cre'),(2*q+spin,'des')
            cf2,sign = string_act(cf1,ops)
            if cf2 is None:
                continue
            ix2 = basis_map[cf2]
            H[ix2,ix1] += M[p,q]*sign
    return H

def eri2MB(eri,basis,basis_map,spin1,spin2,thresh=1e-6):
    H = np.zeros((len(basis),)*2)
    for (p,r,q,s) in itertools.product(range(eri.shape[0]),repeat=4):
        if np.absolute(eri[p,r,q,s])<thresh:
            continue
        for ix1,cf1 in enumerate(basis):
            ops = (2*p+spin1,'cre'),(2*q+spin2,'cre'),(2*s+spin2,'des'),(2*r+spin1,'des')
            cf2,sign = string_act(cf1,ops)
            if cf2 is None:
                continue
            ix2 = basis_map[cf2]
            H[ix2,ix1] += eri[p,r,q,s]*sign
    return H

def bcs2MB(A,B,u,v,basis=None,basis_map=None): 
    nsite,npair = A.shape
    if basis is None:
        basis = all_bitstrings_list(nsite) 
    if basis_map is None:
        basis_map = {det:ix for ix,det in enumerate(basis)}
    nbasis = len(basis)
    def apply_cre(i,psi,A):
        return apply_a_dag_dense_sign(psi,basis,A[:,i],nsite,det_to_index=basis_map)
    
    psi = np.zeros(nbasis)
    psi[basis_map[0]] = 1.
    for k in range(npair//2-1,-1,-1):
        psi_v = apply_cre(2*k+1,psi,A)
        psi_v = apply_cre(2*k,psi_v,A)
        psi = u[2*k]*psi + v[2*k]*psi_v
    nocc = B.shape[1]
    for k in range(nocc-1,-1,-1):
        psi = apply_cre(k,psi,B)
    return psi,basis,basis_map

def det2MB(B,basis=None,basis_map=None,det=0,order=1):
    nsite,nocc = B.shape
    if basis is None:
        basis = all_bitstrings_list(nsite) 
    if basis_map is None:
        basis_map = {det:ix for ix,det in enumerate(basis)}
    nbasis = len(basis)
    def apply_cre(i,psi,A):
        return apply_a_dag_dense_sign(psi,basis,A[:,i],nsite,det_to_index=basis_map)
    psi = np.zeros(nbasis)
    psi[basis_map[det]] = 1.
    ks = range(nocc) if order==1 else range(nocc-1,-1,-1)
    for k in ks:
        psi = apply_cre(k,psi,B)
    return psi,basis,basis_map

def hubbard2MB(h1e,U,symmetry='u11',nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        if symmetry=='u11':
            basis = get_all_configs_u11((nsite,nsite),nelecs)
        elif symmetry=='u1':
            basis = get_all_configs_u1(2*nsite,sum(nelecs))
        else:
            raise NotImplementedError
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,basis_map,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,basis_map,1,thresh=thresh)
    for ix,cf in enumerate(basis):
        H[ix,ix] += U*count_double_occupancy(cf,nsite)
    return H,basis,basis_map

def chol2MB(h1e,chol=None,eri=None,symmetry='u11',nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        if symmetry=='u11':
            basis = get_all_configs_u11((nsite,nsite),nelecs)
        elif symmetry=='u1':
            basis = get_all_configs_u1(2*nsite,sum(nelecs))
        else:
            raise NotImplementedError
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    v0 = 0.
    if chol is not None:
        v0 = .5*xp.einsum('npr,nrs->ps',chol,chol) 

    H = quadratic2MB(h1e-v0,basis,basis_map,0,thresh=thresh)
    H += quadratic2MB(h1e-v0,basis,basis_map,1,thresh=thresh)
    if chol is None:
        for s1,s2 in itertools.product((0,1),repeat=2):
            H += .5 * eri2MB(eri,basis,basis_map,s1,s2,thresh=thresh)
        return H,basis,basis_map

    if eri is None:
        for i,L in enumerate(chol):
            L_ = quadratic2MB(L,basis,basis_map,0,thresh=thresh)
            L_ += quadratic2MB(L,basis,basis_map,1,thresh=thresh)
            H += .5 * np.dot(L_,L_)
        return H,basis,basis_map

def test_norm(D1,D2,thresh=1e-10):
    if D1 is None:
        return
    norm = np.linalg.norm(D1)
    dnorm = np.linalg.norm(D1-D2) 
    if norm<thresh:
        if dnorm>thresh:
            print(dnorm)
            exit()
    else:
        if dnorm/norm>thresh:
            print(dnorm,norm)
            exit()

def test_hamiltonian(ham,nsite,nelecs,h1e,eri=None):
    if eri is None:
        H,basis,basis_map = hubbard2MB(h1e,ham.hubbard_U,nelecs=nelecs)
    else:
       H,basis,basis_map = chol2MB(h1e,eri=eri,symmetry='u11',nelecs=nelecs)
    G1 = np.eye(len(basis))-H/ham.Lambda[-1]
    G2 = get_MB_gf(ham,basis,basis_map)
    #print(G1)
    #print(G2)
    test_norm(G1,G2)
    print('G tested.')

    Us = [ham.get_rotation_matrix(ix) for ix in range(ham.nterms)] 

    occ = [2*i for i in range(nelecs[0])] + [2*i+1 for i in range(nelecs[1])]
    det = get_config_from_occ(occ,nsite*2)
    B = np.random.rand(nsite*2,sum(nelecs))
    psi_det = det2MB(B,basis=basis,basis_map=basis_map,det=det)[0]
    Gpsi_det_1 = np.dot(G1,psi_det)
    Gpsi_det_2 = np.zeros(len(basis))
    for ai,Ui in zip(ham.a,Us):
        UiB = B.copy()
        if Ui[0] is not None:
            UiB[:nsite] = np.dot(Ui[0],B[:nsite])
        if Ui[1] is not None:
            UiB[nsite:] = np.dot(Ui[1],B[nsite:])
        Gpsi_det_2 += ai*det2MB(UiB,basis=basis,basis_map=basis_map,det=det)[0]
    test_norm(Gpsi_det_1,Gpsi_det_2)

if __name__=='__main__':
    from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
    from ipie.utils.linalg import modified_cholesky

    nsite = 5 
    nelecs = 2,0
    method = 2 
    h1e = np.random.rand(nsite,nsite)*2-1
    h1e += h1e.T
    
    print('\ncheck GF decomposition for Hubbard...')
    U = 4 
    ham = HubbardSOR(nsite) 
    at = 10. 
    iprint = 1
    gu = 0.2 
    ham.decompose_h2(U,gu,iprint=iprint)
    ham.decompose_h1(h1e,at,iprint=iprint)
    ham.parse_decomposition()
    test_hamiltonian(ham,nsite,nelecs,h1e)

    print('\ncheck GF decomposition for QC...')
    nchol = 3
    chol = np.random.rand(nchol,nsite,nsite)*2-1
    chol += chol.transpose(0,2,1)
    chol /= 5 
    eri = np.einsum('npr,nqs->prqs',chol,chol) 
    cmax = nsite**2
    M = eri.reshape((nsite**2,)*2)
    print('eri symmetry=',np.linalg.norm(M-M.T))
    chol = modified_cholesky(M,cmax=cmax) 
    chol = chol.reshape(chol.shape[0],nsite,nsite)
    #chol = np.zeros_like(chol)
    #eri = np.zeros_like(eri)
    
    ham = QCSOR(nsite) 
    if method==1:
        ham.decompose_h2_method_1(chol,ai=2.,iprint=iprint,apply_spin_down=(nelecs[1]>0))
    else:
        ham.decompose_h2_method_2(chol,gu,iprint=iprint)
    ham.decompose_h1(h1e,at,iprint=iprint)
    ham.parse_decomposition()
    test_hamiltonian(ham,nsite,nelecs,h1e,eri=eri)
    print('greens function tested')

