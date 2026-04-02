import numpy as np

from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime

from triqs_xca.triqs_solver import TriqsSolver
from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense

def test_block_sparse_self_cons(verbose=False):

    beta = 3.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-12
    w_max = 10.0
    ppsc_tol = 1e-8

    order = 1
    maxiter = 100
    dmft_maxiter = 100
    dmft_tol = 1e-6
    delta_tol = 1e-8

    gf_struct = [['0', 1]]

    from triqs.operators import n
    
    N_op = n('0', 0)
    
    H = -mu * N_op

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

    Delta_w << 0.5 * inverse(iOmega_n - e1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    tau_mesh = Delta_tau.mesh

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)
    
    S.Delta_tau['0'] << Delta_tau # Initial guess

    for iter in range(1, dmft_maxiter+1):

        S.solve(h_int=H, order=order, maxiter=maxiter, tol=ppsc_tol, 
                compress_hybridization=True, update_eta_exact=False)

        print(f'S.S.eta = {S.S.eta:2.2E}')
        print(f'S.S.dmu = {S.S.dmu:2.2E}')

        g_diff = np.max(np.abs(S.G_tau['0'].data - S.Delta_tau['0'].data))
        print(f'iter = {iter}, g_diff = {g_diff:2.2E}')

        S.Delta_tau << S.G_tau

        if g_diff < dmft_tol:
            print(f'DMFT converged in {iter} iterations with g_diff = {g_diff:2.2E}')
            break

    g_S = S.G_tau

    G_S = Gf(mesh=tau_mesh, target_shape=[S.S.G_iaa.shape[-1]]*2)
    G_S.data[:] = S.S.G_iaa.data

    Sigma_S = Gf(mesh=tau_mesh, target_shape=[S.S.G_iaa.shape[-1]]*2)
    Sigma_S.data[:] = S.S.Sigma_iaa


    # -- Block sparse solver

    BSS = BlockSparseSolver(
        H, beta, w_max, eps, gf_struct, 
        #conserved_operators=[],
        )
    
    BSS.Delta_tau['0'] << Delta_tau
 
    for iter in range(1, dmft_maxiter+1):

        BSS.solve(
            max_order=order, tol=ppsc_tol, 
            delta_tol=delta_tol,
            maxiter=maxiter)

        g_BSS = BSS.G_tau

        g_diff = np.max(np.abs(g_BSS['0'].data - BSS.Delta_tau['0'].data))
        print(f'iter = {iter}, g_diff = {g_diff:2.2E}')

        BSS.Delta_tau << g_BSS

        if g_diff < dmft_tol:
            print(f'DMFT (block_sparse) converged in {iter} iterations with g_diff = {g_diff:2.2E}')
            break


    # -- compare

    G_BSS = pseudo_particle_block_gf_to_dense(BSS.pseudo_particle_greens_function(), BSS.atom_diag)

    G_diff = np.max(np.abs(G_BSS.data - G_S.data))
    print(f'G_diff = {G_diff:2.2E}')

    Sigma_BSS = pseudo_particle_block_gf_to_dense(BSS.pseudo_particle_self_energy(), BSS.atom_diag)

    Sigma_diff = np.max(np.abs(Sigma_BSS.data - Sigma_S.data))
    print(f'Sigma_diff = {Sigma_diff:2.2E}')

    g_diff = np.max(np.abs(g_BSS['0'].data - g_S['0'].data))
    print(f'g_diff = {g_diff:2.2E}')

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
        from triqs.gf import make_gf_imtime

        G_BSS_tau = make_gf_imtime(G_BSS, n_tau=400)
        Sigma_BSS_tau = make_gf_imtime(Sigma_BSS, n_tau=400)
        g_BSS_tau = make_gf_imtime(g_BSS, n_tau=400)

        plt.figure(figsize=(8, 9))

        subp = [2, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_BSS_tau, '-')  
        oplotr(G_S)
        plt.ylabel(r'$G(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(Sigma_BSS_tau, '-')
        oplotr(Sigma_S)
        plt.ylabel(r'$\Sigma(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(g_BSS_tau, '-')
        oplotr(g_S)
        plt.ylabel(r'$g(\tau)$')

        plt.tight_layout()
        plt.show()

    np.testing.assert_array_almost_equal(G_BSS.data, G_S.data)
    np.testing.assert_array_almost_equal(Sigma_BSS.data, Sigma_S.data)
    np.testing.assert_array_almost_equal(g_BSS['0'].data, g_S['0'].data)


if __name__ == '__main__':

    test_block_sparse_self_cons(verbose=False)