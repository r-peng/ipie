import numpy as np
import h5py,itertools
from ipie.hamiltonians.sor_base import QCSOR
from ipie.utils.linalg import modified_cholesky
np.set_printoptions(suppress=True,precision=6,threshold=100000,linewidth=100000)

Ns = 21,21,21
ds = [2./(Ni-1) for Ni in Ns]
xs,ys,zs = [np.arange(Ni)*di-1. for Ni,di in zip(Ns,ds)]

mos = np.zeros(Ns+(4,))
mos[:,:,:,0] = xs[:,None,None] 
mos[:,:,:,1] = ys[None,:,None]
mos[:,:,:,2] = zs[None,None,:] 
norm = np.sqrt((mos**2).sum(axis=-1))
ixs = np.nonzero(norm>1.)
mos[ixs] /= norm[ixs][:,None]
norm[ixs] = 1.
mos[:,:,:,3] = np.sqrt(1.-norm**2)

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

def mo2ix(mo):
    r = np.sqrt(1.-mo[3]**2)
    mo /= r
    phi = np.arccos(mo[2])
    theta = np.arctan(mo[1]/mo[0])
    #print(r,phi,theta)

    ir,itheta, = r/dr
    itheta = theta/dtheta
    iphi = phi

    ixs = [(xi-min_i)/dxi for xi,min_i,dxi in zip([r,theta,phi],mins,dx)]

    i,j,k = [int(np.rint(ix)+1e-6) for ix in ixs]
    #print(i,j,k)
    return i,j,k,norm

# pre-compute 
trial = mo_coeff[0,:,0]
Us = np.array([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
Umos = np.einsum('axy,ijky->aijkx',Us,mos)
ix1,ix2,ix3,ix4 = np.nonzero(Umos[:,:,:,:,3]<-1e-6)
Umos_facs = np.sqrt((Umos**2).sum(axis=-1))
Umos /= Umos_facs[:,:,:,:,None]
Umos[ix1,ix2,ix3,ix4] *= -1
print(Umos)
exit()
Umos_facs[minus_idxs] *= -1
print(Umos_facs)
#print(sign)
exit()

r_ = np.sqrt(1.-Umos[:,:,:,:,3]**2)
ir = r_/dr
ir = np.asarray(np.rint(ir)+1e-6,dtype=int)

phi_ = np.arccos(Umos[:,:,:,:,2]/r_)
iphi = phi_/dphi
iphi = np.asarray(np.rint(iphi)+1e-6,dtype=int)
iphi[zero_ixs] = 0
print(phi_)

#theta_ = np.arctan(Umos[:,:,:,:,1]/Umos[:,:,:,:,0])
#itheta = theta_/dtheta
#print(itheta)
#print(iphi)
exit()

Umos_ixs = np.zeros((ham.nterms,Nh),dtype=int) 
for ix1 in range(ham.nterms):
    for ix2,(i,j,k) in enumerate(_flat23d): 
        inew,jnew,knew,fac = mo2ix(Umos[ix1,ix2])
        Umos_ixs[ix1,ix2] = _3d2flat[inew,jnew,knew]
        Umos_facs[ix1,ix2] = fac 
trial_Umos = xp.einsum('a,x,aix->i',ham.a,trial,Umos)
trial_mos = xp.einsum('x,ix->i',trial,mos)

def mat_vec(psi):
    Enum = ham.Lambda[-1]*(1.-np.dot(psi,trial_Umos))
    Edenom = np.dot(psi,trial_mos)
    psi_new = np.zeros(Nh)
    for ix1,ai in enumerate(ham.a):
        psi_new[Umos_ixs[:,ix2]] += ai*psi*Umos_fac[:,ix2]
    return psi_new,Enum/Ednom

def mat_vec_slow(psi):
    psi_new = np.zeros(Nh)
    for ix1,ai in enumerate(ham.a):
        for ix2,moi in enumerate(mos):
            mo_new = xp.dot(Us[ix1],moi)
            i,j,k,fac = mo2ix(mo_new)
            psi_new[_3d2flat[i,j,k]] += fac*psi[ix2]*ai
    return psi_new

psi = np.random.rand(Nh)*2-1.
psi_new1,_ = mat_vec(psi)
psi_new2 = mat_vec_slow(psi)
print('check matvec=',np.linalg.norm(psi_new1-psi_new2),np.linalg.norm(psi_new1))

psi = np.zeros(Ns)
i,j,k,norm = mo2ix(trial)
psi[_3d2flat[i,j,k]] = 1.

