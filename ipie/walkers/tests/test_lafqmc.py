import numpy as np
import scipy,itertools,plum
from ipie.walkers.lafqmc_uhf_walkers import UHFWalkers 
from ipie.walkers.lafqmc_ghf_walkers import GHFWalkers 

importance_sample = True 

def get_data(walkers):
    data = [walkers.phi.copy()]
    if importance_sample:
        data.append(walkers.SCU.copy())
        data.append(walkers.UDU.copy())
    else:
        if 'S' in walkers.buff_names:
            data.append(walkers.S.copy())
        else:
            data.append(walkers.Sa.copy())
            data.append(walkers.Sb.copy())
    return data

def set_data(walkers,data):
    walkers.phi = data[0].copy()
    if importance_sample:
        walkers.SCU = data[1].copy()
        walkers.UDU = data[2].copy()
    else:
        if len(data)==2:
            walkers.S = data[1]
        else:
            walkers.Sa = data[1]
            walkers.Sb = data[2]

def compute_scalar_ovlp(walkers,trial):
    S = walkers.compute_S(trial) 
    if isinstance(S,list):
        det = 1./np.linalg.det(S[0])
        det *= 1./np.linalg.det(S[1])
    else:
        det = 1./np.linalg.det(S)
    return det

def compute_ovlp_ratio_single(ham,ix,walkers,trial):
    old_data = get_data(walkers)
    denom = compute_scalar_ovlp(walkers,trial)

    f = _update_walkers_single(ham,ix,walkers)
    num = compute_scalar_ovlp(walkers,trial)

    set_data(walkers,old_data)
    return num/(denom*f)

@plum.dispatch
def _update_walkers_single(ham,ix,walkers:UHFWalkers):
    nu,nd = walkers.nup,walkers.ndown
    phi = [walkers.phi[:,:,:nu],walkers.phi[:,:,nu:]]
    Us,f = ham.get_rotation_matrix(ix)
    for s in (0,1):
        if phi[s] is None:
            continue
        if Us[s] is None:
            continue
        phi[s] = np.einsum('xy,wyi->wxi',Us[s],phi[s])
    walkers.phi[:,:,:nu] = phi[0]
    walkers.phi[:,:,nu:] = phi[1]
    return f

@plum.dispatch
def _update_walkers_single(ham,ix,walkers:GHFWalkers):
    nb = walkers.nbasis
    Us,f = ham.get_rotation_matrix(ix)
    if Us[0] is not None:
        walkers.phi[:,:nb] = np.einsum('xy,wyi->wxi',Us[0],walkers.phi[:,:nb])
    if Us[1] is not None:
        walkers.phi[:,nb:] = np.einsum('xy,wyi->wxi',Us[1],walkers.phi[:,nb:])
    return f

def update_walkers_slow(ham,ixs,walkers,trial):
    old_data = get_data(walkers)
    denom = compute_scalar_ovlp(walkers,trial)

    f = _update_walkers_slow(ham,ixs,walkers)
    walkers.build(ham,trial,importance=importance_sample)
    num = compute_scalar_ovlp(walkers,trial)
    new_data = get_data(walkers)

    set_data(walkers,old_data)
    return new_data,num/(denom*f)

@plum.dispatch
def _update_walkers_slow(ham,ixs,walkers:UHFWalkers):
    nu,nd = walkers.nup,walkers.ndown
    phi = [walkers.phi[:,:,:nu],walkers.phi[:,:,nu:]]
    f = np.zeros(walkers.nwalkers)
    for w,ix in enumerate(ixs):
        Us,f[w] = ham.get_rotation_matrix(ix)
        for s,Ui in enumerate(Us):
            if phi[s] is None:
                continue
            if Ui is None:
                continue
            phi[s][w] = np.dot(Ui,phi[s][w])
    walkers.phi[:,:,:nu] = phi[0]
    walkers.phi[:,:,nu:] = phi[1]
    return f

@plum.dispatch
def _update_walkers_slow(ham,ixs,walkers:GHFWalkers):
    nb = walkers.nbasis
    f = np.zeros(walkers.nwalkers)
    for w,ix in enumerate(ixs):
        Us,f[w] = ham.get_rotation_matrix(ix)
        if Us[0] is not None:
            walkers.phi[w,:nb] = np.dot(Us[0],walkers.phi[w,:nb])
        if Us[1] is not None:
            walkers.phi[w,nb:] = np.dot(Us[1],walkers.phi[w,nb:])
    return f

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

if __name__=='__main__':
    from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
    from ipie.hamiltonians.generic import GenericRealChol 
    from ipie.utils.linalg import modified_cholesky
    from ipie.walkers.uhf_walkers import UHFWalkers as UHFWalkers_ 
    from ipie.walkers.ghf_walkers import GHFWalkers as GHFWalkers_ 
    from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
    from ipie.trial_wavefunction.single_det import SingleDet as SingleDet_
    from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
    from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF as SingleDetGHF_
    from ipie.qmc.afqmc import AFQMC

    nsite = 5 
    nelecs = 2,1 
    na,nb = nelecs 
    if na>1 and nb==0:
        decomp_type='aa_only'
    elif na==1 and nb==1:
        decomp_type='ab_only'
    else:
        decomp_type='all'
    h1e = np.random.rand(nsite,nsite)*2-1
    h1e += h1e.T

    mpi_handler = None
    nwalker = 3
    initial_walker = np.zeros((nsite,)*2)
    phi = [None] * nwalker 
    phia = [None] * nwalker
    phib = [None] * nwalker
    for i in range(nwalker):
        k = np.random.rand(nsite*2,nsite*2)
        k -= k.T
        phi[i] = scipy.linalg.expm(k)[:,:sum(nelecs)]
        
        k = np.random.rand(nsite,nsite)
        k -= k.T
        phia[i] = scipy.linalg.expm(k)[:,:nelecs[0]]
    
        k = np.random.rand(nsite,nsite)
        k -= k.T
        phib[i] = scipy.linalg.expm(k)[:,:nelecs[1]]
    phi0 = np.array(phi)
    phia0 = np.array(phia)
    phib0 = np.array(phib)
    walkers_types = 'uhf','ghf',

    k = np.random.rand(nsite*2,nsite*2)
    k -= k.T
    phi_ghf = scipy.linalg.expm(k)[:,:sum(nelecs)]

    phi_uhf = [None] * 2
    for s in (0,1):
        k = np.random.rand(nsite,nsite)
        k -= k.T
        phi_uhf[s] = scipy.linalg.expm(k)[:,:nelecs[s]]
    phi_uhf = np.concatenate(phi_uhf,axis=1)
    trial_types = 'uhf','ghf',
    
    for walkers_type,trial_type in itertools.product(walkers_types,trial_types):
        if walkers_type=='ghf' and trial_type=='uhf':
            continue
        print()
        print(f'walkers_type={walkers_type},trial_type={trial_type}')
        if walkers_type=='uhf':
            walkers = UHFWalkers(initial_walker,nelecs[0],nelecs[1],nsite,nwalker,mpi_handler)
            walkers.weights = np.random.rand(nwalker)
            walkers.phi = np.concatenate([phia0,phib],axis=2)

            walkers_ = UHFWalkers_(initial_walker,nelecs[0],nelecs[1],nsite,nwalker,mpi_handler)
            walkers_.phia = phia.copy()
            walkers_.phib = phib.copy()
        else:
            walkers = GHFWalkers(initial_walker,nelecs[0],nelecs[1],nsite,nwalker,mpi_handler)
            walkers.weights = np.random.rand(nwalker)
            walkers.phi = phi0.copy() 

            walkers_ = GHFWalkers_(initial_walker,nelecs[0],nelecs[1],nsite,nwalker,mpi_handler)
            walkers_.phi = phi0.copy() 
        walkers_.weights = walkers.weight.copy()
        
        if trial_type=='uhf':
            trial = SingleDet(phi_uhf,nelecs,nsite)
            trial_ = SingleDet_(phi_uhf,nelecs,nsite)
        else:
            trial = SingleDetGHF(phi_ghf,nelecs,nsite)
            trial_ = SingleDetGHF_(phi_ghf,nelecs,nsite)

        hams = [None] * 2
        generic_real_chols = [None] * 2
        iprint = 1

        U = 4 
        dt = 0.05
        if nelecs[1]>0:
            hams[0] = HubbardSOR(nsite,decomp_type=decomp_type) 
            hams[0].decompose_h2(U,dt,iprint=iprint,trial=trial)
            hams[0].decompose_h1(h1e,dt,iprint=iprint,trial=trial)
            hams[0].parse_decomposition()
            eri = np.zeros((nsite,)*4)
            for i in range(nsite):
                eri[i, i, i, i] = U
            verbose = True 
            chol = modified_cholesky(eri.reshape((nsite**2,)*2),verbose=verbose,cmax=nsite) 
            generic_real_chols[0] = GenericRealChol(np.array([h1e,h1e]),chol.T,0)

        nchol = 3
        chol = np.random.rand(nchol,nsite,nsite)*2-1
        chol += chol.transpose(0,2,1)
        chol /= 2
        eri = np.einsum('npr,nqs->prqs',chol,chol) 
        cmax = nsite**2
        M = eri.reshape((nsite**2,)*2)
        print('eri symmetry=',np.linalg.norm(M-M.T))
        chol = modified_cholesky(M,cmax=cmax) 
        chol = chol.reshape(chol.shape[0],nsite,nsite)
        hams[1] = QCSOR(nsite,decomp_type=decomp_type) 
        hams[1].decompose_h2(chol,dt,iprint=iprint,trial=trial)
        hams[1].decompose_h1(h1e,dt,iprint=iprint,trial=trial)
        hams[1].parse_decomposition()
        chol = chol.reshape(nchol,nsite**2)
        generic_real_chols[1] = GenericRealChol(np.array([h1e,h1e]),chol.T,0)
        for ham,generic_real_chol in zip(hams,generic_real_chols):
            if ham is None:
                continue
            print(type(ham))
        
            if walkers_type==trial_type:
                if trial_type=='uhf':
                    trial_.half_rotate(generic_real_chol)
                afqmc = AFQMC.build(nelecs,generic_real_chol,trial_,walkers=walkers_,num_walkers=nwalker,num_steps_per_block=1,num_blocks=1,timestep=0.001)
                afqmc.setup_estimators(None,None)
            
            trial.build(ham,conjugate=True)
            walkers.build(ham,trial,importance=False)
            if walkers_type==trial_type:
                eloc,e1,e2 = walkers.local_energy(ham,trial)
                E = np.dot(eloc,walkers.weight)/sum(walkers.weight)
                e1 = np.dot(e1,walkers.weight)/sum(walkers.weight)
                e2 = np.dot(e2,walkers.weight)/sum(walkers.weight)
                print('E,E1,E2=',E,e1,e2)

            walkers.build(ham,trial,importance=importance_sample)
            walkers.has_E12 = False
            if walkers_type==trial_type:
                eloc,e1,e2 = walkers.local_energy(ham,trial)
                E = np.dot(eloc,walkers.weight)/sum(walkers.weight)
                e1 = np.dot(e1,walkers.weight)/sum(walkers.weight)
                e2 = np.dot(e2,walkers.weight)/sum(walkers.weight)
                print('E,E1,E2=',E,e1,e2)
            #continue

            if importance_sample:
                ovlp_ratio1 = np.zeros((ham.nterms,walkers.nwalkers))
                for ix in range(ham.nterms):
                    ovlp_ratio1[ix] = compute_ovlp_ratio_single(ham,ix,walkers,trial)
                ovlp_ratio2 = walkers.compute_ovlp_ratio(ham)
                #print('ovlp ratio')
                #print(ovlp_ratio1.T)
                #print(ovlp_ratio2.T)
                test_norm(ovlp_ratio1,ovlp_ratio2)

            niter = ham.nterms // walkers.nwalkers + 1
            start = 0
            for i in range(niter):
                stop = start + nwalker
                ixs = np.arange(start,stop) % ham.nterms 
                #print('ixs=',ixs)
                #print('keys=',[ham.ix2key[ix] for ix in ixs])

                data1,b1 = update_walkers_slow(ham,ixs,walkers,trial)

                data_old = get_data(walkers)
                ham.parse_samples(ixs)
                b2 = None if importance_sample else np.ones(walkers.nwalkers)
                b2 = walkers.update_walkers(ham,trial,b=b2)
                if b2 is not None:
                    #print(b1)
                    #print(b2)
                    test_norm(b1,b2)
                data2 = get_data(walkers)
                set_data(walkers,data_old)
                for item1,item2 in zip(data1,data2):
                    #print(item1.shape)
                    #print(item1)
                    #print(item2)
                    test_norm(item1,item2)
                start = stop

print('all tests completed.')
