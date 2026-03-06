
import numpy as np

from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime

from triqs_xca.triqs_solver import TriqsSolver

from triqs_xca.block_sparse_solver import BlockSparseSolver

from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense

def test_block_sparse_self_cons():

    beta = 3.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-12
    w_max = 10.0 * beta
    tol = 1e-6

    order = 1
    maxiter = 2

    gf_struct = [['0', 1]]
    fops = [ ('0', 0) ]

    #gf_struct = [['0', 1], ['1', 1]]
    #fops = [ ('0', 0), ('1', 0) ]

    from triqs.operators import n
    
    #N_op = n('0', 0) + n('1', 0)
    N_op = n('0', 0)
    
    H = -mu * N_op

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

    Delta_w << 2.0 * inverse(iOmega_n - e1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)
    S.Delta_tau['0'] << Delta_tau
    #S.Delta_tau['1'] << Delta_tau
    S.solve(h_int=H, order=order, maxiter=maxiter, tol=tol, 
            compress_hybridization=True, update_eta_exact=False)
    print(f'S.S.eta = {S.S.eta:2.2E}')
    print(f'S.S.dmu = {S.S.dmu:2.2E}')


    BSS = BlockSparseSolver(H, beta, w_max, eps, gf_struct, conserved_operators=[])
    
    BSS.Delta_tau['0'] << Delta_tau
    #BSS.Delta_tau['1'] << Delta_tau

    from triqs.gf import make_gf_dlr_imfreq
    Delta_iw = make_gf_dlr_imfreq(BSS.Delta_tau) # BlockGf
    
    BSS.fit_hybridization(tol=tol)

    print(f'BSS.hyb.poles = {BSS.hyb.poles}')
    print(f'BSS.hyb.coefficients =\n{BSS.hyb.coefficients}')

    BSS.init_diagram_evaluator()
    BSS.solve(max_order=order, tol=tol, maxiter=maxiter)

    # -- compare

    G_BSS = pseudo_particle_block_gf_to_dense(BSS.pseudo_particle_greens_function(), BSS.ad)
    G_S = G_BSS.copy()
    G_S.data[:] = S.S.G_iaa.data

    G_diff = np.max(np.abs(G_BSS.data - G_S.data))
    print(f'G_diff = {G_diff:2.2E}')

    Sigma_BSS = pseudo_particle_block_gf_to_dense(BSS.Sigma, BSS.ad)
    Sigma_S = Sigma_BSS.copy()
    Sigma_S.data[:] = S.S.Sigma_iaa

    Sigma_diff = np.max(np.abs(Sigma_BSS.data - Sigma_S.data))
    print(f'Sigma_diff = {Sigma_diff:2.2E}')

    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
    from triqs.gf import make_gf_imtime

    G_BSS_tau = make_gf_imtime(G_BSS, n_tau=400)
    Sigma_BSS_tau = make_gf_imtime(Sigma_BSS, n_tau=400)

    plt.figure(figsize=(8, 9))

    subp = [2, 2, 1]

    plt.subplot(*subp); subp[-1] += 1
    oplotr(G_BSS_tau, '-')  
    oplotr(G_S)

    plt.subplot(*subp); subp[-1] += 1
    oplotr(Sigma_BSS_tau, '-')
    oplotr(Sigma_S)

    plt.show()


if __name__ == '__main__':

    test_block_sparse_self_cons()