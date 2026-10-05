import numpy as np
import plum,h5py
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.walkers.base_walkers import BaseWalkers 
from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
from ipie.utils.backend import to_host,cast_to_device
from ipie.utils.backend import arraylib as xp

def qr(phi,thresh=1e-3):
    Q,R = xp.linalg.qr(phi,mode='reduced')
    Rdiag = xp.einsum('wii->wi',R)

    Rabs = xp.fabs(Rdiag)
    assert Rabs[Rabs<thresh].size==0

    sign = xp.sign(Rdiag)
    Q *= sign[:,None,:]
    return Q

def update_phi(C,v,d):
    vC = xp.einsum('wxr,wxi->wri',v,C)
    if len(d.shape)==2:
        dvC = d[:,:,None]*vC
    else:
        dvC = xp.einsum('wrs,wsi->wri',d,vC)
    C += xp.einsum('wxr,wri->wxi',v[:,:,::-1],dvC)
    return C,dvC

class UHFWalkers(BaseWalkers):

    def __init__(
        self,
        initial_walker: np.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        mpi_handler,
        write_filepath=None,
        write_restart=False,
        write_freq=None,
        write_time=None,
        verbose: bool = False,
    ):
        assert len(initial_walker.shape) == 2
        self.nup = nup
        self.ndown = ndown
        self.nelec = nup+ndown
        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        super().__init__(
            nwalkers,
            write_filepath=write_filepath,
            write_restart=write_restart,
            write_freq=write_freq,
            write_time=write_time,
            verbose=verbose,
        )

        self.phi = xp.array([initial_walker.copy() for iw in range(self.nwalkers)])
        self.measure_sign = False

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def get_phi(self):
        nu = self.nup
        return [self.phi[:,:,:nu],self.phi[:,:,nu:]]

    def set_phi(self,phi):
        nu = self.nup
        self.phi[:,:,:nu] = phi[0]
        self.phi[:,:,nu:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDet,set_attribute=True,set_buff=False):
        phi = self.get_phi()
        BC = [xp.einsum('xi,wxj->wij',Bi,Ci) for Bi,Ci in zip(trial.psi,phi)] 
        S = [xp.linalg.inv(Si) for Si in BC]
        if set_attribute:
            self.Sa,self.Sb = S
        if set_buff:
            self.buff_names = ['Sa','Sb']
        return S
    
    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF,set_attribute=True,set_buff=False):
        phi = self.get_phi()
        B = [trial.psi[:self.nbasis],trial.psi[self.nbasis:]]
        BC = [xp.einsum('xi,wxj->wij',Bi,Ci) for Bi,Ci in zip(B,phi)] 
        BC = xp.concatenate(BC,axis=2)
        S = xp.linalg.inv(BC)
        if set_attribute:
            self.S = S
        if set_buff:
            self.buff_names = ['S']
        return S

    def build(self,trial):
        self.compute_S(trial,set_buff=True)
        self.buff_names += ['phi','weight','phase','unscaled_weight','hybrid_energy']

        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = np.zeros(self.buff_size, dtype=np.complex128)

    def update_walkers(self,hamiltonian,trial,b=None):
        self.itm = dict()
        self.has_E12 = False 
        for key,ixs in hamiltonian.samples.items():
            w,i = ixs['w'],ixs['i']
            p,d,v = hamiltonian.get_batch_info(key,i)

            size_key,(bix,spin) = key
            dvC = self.update_phi(spin,w,v,d)

            if spin==(0,1):
                b = self.update_ovlp_2(key,w,p,dvC,trial,b)
            else:
                b = self.update_ovlp_1(key,w,p,dvC,trial,b)
            #print(b[w],f)
            #b[w] /= f 
            #print(b[w])
        return b

    def update_phi(self,spin,w,v,d):
        phi = self.get_phi()
        if spin==(0,1):
            d = [d[:,:1],d[:,1:]]
            v = [v[:,:,:1],v[:,:,1:]]
            dvC = [None] * 2
            for s in (0,1):
                phi[s][w],dvC[s] = update_phi(phi[s][w],v[s],d[s])
        else:
            s = spin[0]
            phi[s][w],dvC = update_phi(phi[s][w],v,d)
        self.set_phi(phi)
        return dvC

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,dvC,trial:SingleDet,b):
        size_key,(bix,spin) = key
        s = spin[0]

        Bw = trial.get_Bv(size_key,bix,s,p[:,::-1])
        S = [self.Sa,self.Sb][s]
        SBw = xp.einsum('wij,wjr->wir',S[w],Bw)
        dvCS = xp.einsum('wri,wij->wrj',dvC,S[w])

        M = xp.eye(p.shape[1])[None,:,:] + xp.einsum('wri,wis->wrs',dvC,SBw)
        if b is not None:
            b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M)
        right = xp.einsum('wrs,wsj->wrj',M,dvCS)
        S[w] -= xp.einsum('wir,wrj->wij',SBw,right)
        if s==0:
            self.Sa = S
        else:
            self.Sb = S
        return b 

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,dvC,trial:SingleDetGHF,b):
        size_key,(bix,spin) = key
        s = spin[0]
        Bw = trial.get_Bv(size_key,bix,s,p[:,::-1])
        SBw = xp.einsum('wij,wjr->wir',self.S[w],Bw)
        if s==0:
            dvCS = xp.einsum('wri,wij->wrj',dvC,self.S[w,:self.nup])
        else:
            dvCS = xp.einsum('wri,wij->wrj',dvC,self.S[w,self.nup:])

        if s==0:
            M = xp.einsum('wri,wis->wrs',dvC,SBw[:,:self.nup])
        else:
            M = xp.einsum('wri,wis->wrs',dvC,SBw[:,self.nup:])
        M = xp.eye(p.shape[1])[None,:,:] + M
        if b is not None:
            b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M)
        right = xp.einsum('wrs,wsj->wrj',M,dvCS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SBw,right)
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,dvC,trial:SingleDet,b):
        size_key,(bix,spin) = key
        p = [p[:,:1],p[:,1:]]
        Bw = [trial.get_Bv(size_key,bix,s,p[s]) for s in (0,1)]
        S = [self.Sa,self.Sb]
        for s in (0,1):
            SBw = xp.einsum('wij,wj->wi',S[s][w],Bw[s][:,:,0])
            dvCS = xp.einsum('wi,wij->wj',dvC[s][:,0],S[s][w])

            M = 1. + xp.einsum('wi,wi->w',dvC[s][:,0],SBw)
            if b is not None:
                b[w] *= M

            right = (1./M)[:,None]*dvCS
            S[s][w] -= xp.einsum('wi,wj->wij',SBw,right)

        self.Sa,self.Sb = S
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,dvC,trial:SingleDetGHF,b):
        size_key,(bix,spin) = key
        p = [p[:,:1],p[:,1:]]
        Bw = [trial.get_Bv(size_key,bix,s,p[s]) for s in (0,1)]
        Bw = xp.concatenate(Bw,axis=2)
        SBw = xp.einsum('wij,wjr->wir',self.S[w],Bw)
        dvCS = [None] * 2
        dvCS[0] = xp.einsum('wri,wij->wrj',dvC[0],self.S[w,:self.nup])
        dvCS[1] = xp.einsum('wri,wij->wrj',dvC[1],self.S[w,self.nup:])
        dvCS = xp.concatenate(dvCS,axis=1)

        M = [None] * 2 
        M[0] = xp.einsum('wri,wis->wrs',dvC[0],SBw[:,:self.nup])
        M[1] = xp.einsum('wri,wis->wrs',dvC[1],SBw[:,self.nup:])
        M = xp.eye(2)[None,:,:] + xp.concatenate(M,axis=1)
        if b is not None:
            b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M)
        right = xp.einsum('wrs,wsj->wrj',M,dvCS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SBw,right)
        return b 

    def reortho(self,trial):
        phi = self.get_phi()
        for s in (0,1):
            phi[s] = qr(phi[s])
        self.set_phi(phi)
        if 'S' in self.buff_names or 'Sa' in self.buff_names:
            self.compute_S(trial)

    def save(self,comm,dirname):
        RANK,SIZE = comm.rank,comm.size
        if RANK>0:
            obj = to_host(self.phi),to_host(self.weight)
            comm.send(obj,0)
            return
        phi = [to_host(self.phi)] + ([None] * (SIZE-1))
        weights = [to_host(self.weight)] + ([None] * (SIZE-1))
        for r in range(1,SIZE):
            phi[r],weights[r] = comm.recv(source=r)
        with h5py.File(f'{dirname}/walkers.hdf5','w') as f:
            f.create_dataset('phi',data=np.concatenate(phi,axis=0))
            f.create_dataset('weights',data=np.concatenate(weights,axis=0))
    
    def load(self,comm,dirname):
        with h5py.File(f'{dirname}/walkers.hdf5','r') as f:
            phi = f['phi'][:]
            weights = f['weights'][:]
        RANK,SIZE = comm.rank,comm.size
    
        nw = weights.size
        b,r = nw//SIZE,nw%SIZE
        counts = np.array([b]*SIZE)
        if r>0:
            counts[:r] += 1
        counts = np.cumsum(counts)
        start = 0 if RANK==0 else counts[RANK-1]
        stop = counts[RANK]
        print(f'RANK={RANK},start={start},stop={stop}')
        self.phi = np.asarray(phi[start:stop])
        self.weight = np.asarray(weights[start:stop])
        #_check_nan(walkers.phi,'phi','loaded')
        #nu = walkers.nup
        #phi = walkers.phi[:,:,:nu]
        #ovlp = xp.einsum('wxi,wxj->wij',phi,phi)
        #print(np.linalg.norm(ovlp-xp.eye(nu)[None,:,:]))
        #phi = walkers.phi[:,:,nu:]
        #ovlp = xp.einsum('wxi,wxj->wij',phi,phi)
        #print(np.linalg.norm(ovlp-xp.eye(nu)[None,:,:]))
        #exit()
    
    def reortho_batched(self):
        pass

    def compute_CS(self):
        phi = self.get_phi()
        S = self.S if 'S' in self.buff_names else [self.Sa,self.Sb]
        if isinstance(S,list):
            return [xp.einsum('wxi,wij->wxj',Ci,Si) for Ci,Si in zip(phi,S)]
        else:
            nb = self.nbasis
            nu = self.nup
            Sa = xp.einsum('wxi,wij->wxj',phi[0],S[:,:nu])
            Sb = xp.einsum('wxi,wij->wxj',phi[1],S[:,nu:])
            return [Sa,Sb]

    def local_energy(self,ham,trial):
        CS = self.compute_CS()
        E1 = [xp.einsum('wxi,ix->w',CSi,Bi) for CSi,Bi in zip(CS,trial.Bh1)]
        E1 = E1[0]+E1[1]

        E2 = 0.
        spin_pairs = [(s1,s2) for s1 in (0,1) for s2 in (0,1)]
        cross = 'S' in self.buff_names
        for size_key,sg in ham.size_groups.items():
            BX = trial.BX[size_key]
            XCS = [xp.einsum('dxp,wxi->dwpi',sg.basis,CSi) for CSi in CS]
            for bix,tag in enumerate(sg.integral_tags):
                if tag=='h1':
                    continue
                W = sg.integrals[bix]
                rho_a = xp.einsum('wpi,ip->wp',XCS[0][bix],BX[0][bix])
                rho_b = xp.einsum('wpi,ip->wp',XCS[1][bix],BX[1][bix])
                if tag=='hubbard':
                    E2 += (W[None,:]*rho_a*rho_b).sum(axis=1)
                    if cross:
                        rho_ab = xp.einsum('wpi,ip->wp',XCS[0][bix],BX[1][bix])
                        rho_ba = xp.einsum('wpi,ip->wp',XCS[1][bix],BX[0][bix])
                        E2 -= (W[None,:]*rho_ab*rho_ba).sum(axis=1)
                if tag=='thc':
                    rho_ = rho_a + rho_b
                    E2 += .5*xp.einsum('ab,wa,wb->w',W,rho_,rho_)

                    S = sg.overlaps[bix]
                    Da = xp.einsum('wpi,iq->wpq',XCS[0][bix],BX[0][bix])
                    Db = xp.einsum('wpi,iq->wpq',XCS[1][bix],BX[1][bix])
                    if S is None:
                        E1 += .5*xp.einsum('a,waa->w',xp.diag(W),Da+Db)
                    else:
                        E1 += .5*xp.einsum('ab,wab->w',W*S,Da+Db)
                    E2 -= .5*xp.einsum('ab,wab,wba->w',W,Da,Da)
                    E2 -= .5*xp.einsum('ab,wab,wba->w',W,Db,Db)
                    if cross:
                        Dab = xp.einsum('wpi,iq->wpq',XCS[0][bix],BX[1][bix])
                        Dba = xp.einsum('wpi,iq->wpq',XCS[1][bix],BX[0][bix])
                        E2 -= .5*xp.einsum('ab,wab,wba->w',W,Dab,Dba)
                        E2 -= .5*xp.einsum('ab,wab,wba->w',W,Dba,Dab)
        return E1+E2,E1,E2

    def _measure_sign(self,hamiltonian,trial):
        self.compute_density(hamiltonian,trial,set_buff=False)
        ovlp = self.compute_ovlp_ratio(hamiltonian)
        g = hamiltonian.a[:,None] * ovlp
        gsum = g.sum(axis=0)
        bsum = xp.fabs(g).sum(axis=0)

        b_plus = g.copy()
        xp.clip(b_plus, a_min=0.0, a_max=None, out=b_plus)  
        b_plus = b_plus.sum(axis=0)

        b_minus = g.copy()
        xp.clip(b_minus, a_min=None, a_max=0.0, out=b_minus)  
        b_minus = b_minus.sum(axis=0)
        b_minus *= -1

        err = xp.linalg.norm(b_plus+b_minus-bsum)
        if err>1e-10:
            print(err)
            exit()
        err = xp.linalg.norm(b_plus-b_minus-gsum)
        if err>1e-10: 
            print(err)
            exit()
        f = b_minus / bsum
        s = xp.fabs(gsum) / bsum
        return f,s
