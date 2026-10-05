import numpy as np
import plum
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
from ipie.walkers.lafqmc_uhf_walkers import (
        UHFWalkers,
        qr,
)

class GHFWalkers(UHFWalkers):

    def get_phi(self):
        phi = [self.phi[:,:self.nbasis],self.phi[:,self.nbasis:]]
        return phi

    def set_phi(self,phi):
        self.phi[:,:self.nbasis] = phi[0]
        self.phi[:,self.nbasis:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDet,**kwargs):
        raise NotImplementedError

    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF,set_attribute=True,set_buff=False):
        CB = xp.einsum('xi,wxj->wij',trial.psi,self.phi)
        S = xp.linalg.inv(CB)
        if set_attribute:
            self.S = S
        if set_buff:
            self.buff_names = ['S']
        return S

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,dvC,trial:SingleDet,b):
        raise NotImplementedError

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,dvC,trial:SingleDetGHF,b):
        size_key,(bix,spin) = key
        s = spin[0]

        Bv = trial.get_Bv(size_key,bix,s,p[:,::-1])
        SBv = xp.einsum('wij,wjr->wir',self.S[w],Bv)
        dvCS = xp.einsum('wri,wij->wrj',dvC,self.S[w])

        M = xp.eye(p.shape[1])[None,:,:] + xp.einsum('wri,wis->wrs',dvC,SBv)
        if b is not None:
            b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M)
        right = xp.einsum('wrs,wsj->wrj',M,dvCS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SBv,right)
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,dvC,trial:SingleDet,b):
        raise NotImplementedError

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,dvC,trial:SingleDetGHF,b):
        size_key,(bix,spin) = key
        p = [p[:,:1],p[:,1:]]
        Bv = xp.concatenate([trial.get_Bv(size_key,bix,s,p[s]) for s in (0,1)],axis=2)
        SBv = xp.einsum('wij,wjr->wir',self.S[w],Bv)
        dvC = xp.concatenate(dvC,axis=1)
        dvCS = xp.einsum('wri,wij->wrj',dvC,self.S[w])

        M = xp.eye(2)[None,:,:] + xp.einsum('wri,wis->wrs',dvC,SBv)
        if b is not None:
            b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M)
        right = xp.einsum('wrs,wsj->wrj',M,dvCS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SBv,right)
        return b 

    def reortho(self,trial):
        self.phi = qr(self.phi)
        if 'S' in self.buff_names:
            self.compute_S(trial)

    @plum.dispatch
    def compute_CS(self,trial:SingleDet):
        raise NotImplementedError

    @plum.dispatch
    def compute_CS(self,trial:SingleDetGHF):
        phi = self.get_phi()
        return [xp.einsum('wxi,wij->wxj',Ci,self.S) for Ci in phi]

    def _load_phi(self,phi):
        self.phi = xp.asarray(phi)
