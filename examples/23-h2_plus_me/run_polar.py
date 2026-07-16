import numpy as np
import h5py,itertools,time
from ipie.hamiltonians.sor_base import QCSOR
from ipie.utils.linalg import modified_cholesky
np.set_printoptions(suppress=True,precision=6)

N = 10
Nr,Ntheta,Nphi = 11,N*4,N*2+1
dr,dtheta,dphi = 1./(Nr-1),2*np.pi/Ntheta,np.pi/(Nphi-1)
rs,thetas,phis = [np.arange(Nx)*dx for Nx,dx in zip([Nr,Ntheta,Nphi],[dr,dtheta,dphi])]
print(rs)
print(thetas)
print(phis)

print('computing all mos...')
t0 = time.time()
mos = np.zeros((Nr,Ntheta,Nphi,4))
mos[:,:,:,0] = np.einsum('i,j,k->ijk',rs,np.cos(thetas),np.sin(phis)) 
mos[:,:,:,1] = np.einsum('i,j,k->ijk',rs,np.sin(thetas),np.sin(phis)) 
mos[:,:,:,2] = np.einsum('i,k->ik',rs,np.cos(phis))[:,None,:] 
mos[:,:,:,3] = np.sqrt(1.-rs**2)[:,None,None]
flat2idx = list(itertools.product(range(Nr),range(Ntheta),range(Nphi)))
idx2flat = {(i,j,k):ix for ix,(i,j,k) in enumerate(flat2idx)}
Nh = Nr*Ntheta*Nphi
mos = mos.reshape((Nh,4))
print('computing mo time=',time.time()-t0)

R = 1.1
xc = 'b3lyp'
print(f'############### R={R:.1f} #################')
with h5py.File(f'integrals/R{R:.1f}_{xc}.h5','r') as f:
    h1e = f['hcore'][:]
    eri = f['eri'][:]
    mo_coeff = f['mo_coeff'][:]

nelecs = 1,0
nsite = eri.shape[0] 
cmax = nsite**2
M = eri.reshape((nsite**2,)*2)
print('eri symmetry=',np.linalg.norm(M-M.T))
chol = modified_cholesky(M,cmax=cmax) 
chol = chol.reshape(chol.shape[0],nsite,nsite)
ham = QCSOR(apply_spin_down=False) 

iprint = 1 
dt = 0.01
at = 1./dt 
ham.decompose_h1(h1e,at,iprint=iprint)
ai = 1./np.sqrt(dt) 
ai = 5.
ham.decompose_h2(chol,ai,iprint=iprint)
ham.parse_decomposition()

def mo2ix(mo,thresh=1e-8):
    norm = np.linalg.norm(mo)
    mo /= norm
    if mo[3]<-thresh:
        norm *= -1
        mo *= -1

    r = np.sqrt(1.-mo[3]**2)
    if r<thresh:
        return idx2flat[0,0,0],norm
    ir = int(np.rint(r/dr)+1e-6)
    #print(ir,r/dr,r,dr,mo)
    #exit()

    mo /= r
    if mo[2]<-1+thresh:
        return idx2flat[ir,0,Nphi-1],norm
    if mo[2]>1.-thresh:
        return idx2flat[ir,0,0],norm
    phi = np.arccos(mo[2])
    iphi = int(np.rint(phi/dphi)+1e-6)

    if np.fabs(mo[0])<thresh:
        if mo[1]>0.:
            return idx2flat[ir,N,iphi],norm
        else:
            return idx2flat[ir,3*N,iphi],norm
    if np.fabs(mo[1])<thresh:
        if mo[0]>0:
            return idx2flat[ir,0,iphi],norm
        else:
            return idx2flat[ir,2*N,iphi],norm
    theta = np.arctan(mo[1]/mo[0])
    if theta>0:
        if mo[0]<0:
            theta += np.pi
    if theta<0:
        if mo[0]<0:
            theta += np.pi
        else:
            theta += 2*np.pi
    itheta = int(np.rint(theta/dtheta)+1e-6)
    if itheta==Ntheta:
        itheta = 0
    return idx2flat[ir,itheta,iphi],norm

#test_set = [np.array([x,y,z,w]) for x,y,z,w in itertools.product((-1.,0.,1.),repeat=4)]
#for vec in test_set:
#    norm = np.linalg.norm(vec)
#    if norm<1e-6:
#        continue
#    vec /= norm 
#    print(vec)
#    ix,norm = mo2ix(vec)
#    print(flat2idx[ix],norm)
#exit()

# pre-compute 
print('precomputing Umos...')
t0 = time.time()
trial = mo_coeff[0,:,0]
Us = np.array([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
Umos = np.einsum('axy,iy->aix',Us,mos)
print('computing Umo time=',time.time()-t0)
try:
    Umos_ixs = np.load('integrals/R{R:.1f}_{xc}_Umos_ixs_{Nr}_{N}.npy')
    Umos_fac = np.load('integrals/R{R:.1f}_{xc}_Umos_fac_{Nr}_{N}.npy')
except FileNotFoundError:
    print('precomputing index and fac maps...')
    t0 = time.time()
    Umos_ixs = np.zeros((ham.nterms,Nh),dtype=int)
    Umos_fac = np.zeros((ham.nterms,Nh))
    for ix1 in range(ham.nterms):
        for ix2 in range(Nh):
            #print(ix1,ix2)
            ix,fac = mo2ix(Umos[ix1,ix2])
            Umos_ixs[ix1,ix2] = ix
            Umos_fac[ix1,ix2] = fac
    np.save('integrals/R{R:.1f}_{xc}_Umos_ixs_{Nr}_{N}.npy',Umos_ixs) 
    np.save('integrals/R{R:.1f}_{xc}_Umos_fac_{Nr}_{N}.npy',Umos_fac) 
    print('computing idx map time=',time.time()-t0)

#trial_Umos = np.einsum('a,x,aix->i',ham.a,trial,Umos)
trial_mos = np.einsum('x,ix->i',trial,mos)

def mat_vec(psi):
    psi_new = np.zeros(Nh)
    for ix1,ai in enumerate(ham.a):
        for ix2 in range(Nh):
            psi_new[Umos_ixs[ix1,ix2]] += ai*psi[ix2]*Umos_fac[ix1,ix2]
    return psi_new

def mat_vec_slow(psi):
    psi_new = np.zeros(Nh)
    for ix1,ai in enumerate(ham.a):
        for ix2,moi in enumerate(mos):
            #print(ix1,ix2)
            mo_new = np.dot(Us[ix1],moi)
            ix,fac = mo2ix(mo_new)
            psi_new[ix] += fac*psi[ix2]*ai
    return psi_new

check = True
check = False
if check:
    t0 = time.time()
    print('computing fast matvec...')
    psi = np.random.rand(Nh)*2-1.
    psi_new1 = mat_vec(psi)
    print('computing slow matvec...')
    psi_new2 = mat_vec_slow(psi)
    print('check matvec=',np.linalg.norm(psi_new1-psi_new2),np.linalg.norm(psi_new1))
    print('check vector time=',time.time()-t0)
    exit()

psi = np.zeros(Nh)
ix,norm = mo2ix(trial)
psi[ix] = 1.
Nstep = 200
E = []
for i in range(Nstep):
    Gpsi = mat_vec(psi) 
    Ei = np.dot(trial_mos,Gpsi)/np.dot(trial_mos,psi)
    Ei = ham.Lambda[-1]*(1.-Ei)
    E.append(Ei)
    print(i,Ei)
    psi = Gpsi
