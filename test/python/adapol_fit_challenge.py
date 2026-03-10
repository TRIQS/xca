import numpy as np

from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime, make_gf_dlr

from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense


def from_blockgf_to_dense(G):

    for b, g in G:
        assert( len(g.target_shape) == 2)
        assert( g.target_shape[0] == g.target_shape[1] )

    norb = sum([ g.target_shape[0] for b, g in G ])
    
    G_dense = Gf(mesh=G.mesh, target_shape=[norb]*2)
    
    sidx = 0
    for b, g in G:
        size = g.target_shape[0]
        G_dense.data[:, sidx:sidx+size, sidx:sidx+size] = g.data
        sidx += size

    return G_dense


def adapol_fit_test():

    beta = 3.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-12
    w_max = 10.0 * beta

    order = 1
    maxiter = 100
    ppsc_tol = 1e-10
    adapol_tol = ppsc_tol * 0.1

    gf_struct = [['0', 1]]

    from triqs.operators import n
    H = -mu * n('0', 0)

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

    Delta_w << 0.5 * inverse(iOmega_n - e1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    # -- Block sparse solver

    BSS = BlockSparseSolver(H, beta, w_max, eps, gf_struct)
    BSS.Delta_tau['0'] << Delta_tau
    BSS.solve(max_order=order, tol=ppsc_tol, maxiter=maxiter)
 
    g_BSS = BSS.G_tau

    BSS.Delta_tau << BSS.G_tau # Bethe self-consistency

    from adapol.fit_utils_dlr import polefitting_dlr

    Delta_dlr = make_gf_dlr(BSS.Delta_tau)
    Delta_dlr_dense = from_blockgf_to_dense(Delta_dlr)
    w_dlr = np.array([ float(x) for x in Delta_dlr.mesh ])
    
    print('='*72)
    print('='*72)
    print('--> Calling adapol.fit_utils_dlr.polefitting_dlr')
    print('='*72)
    print('='*72)

    pole_weights, poles, fit_error = polefitting_dlr(
        Delta_dlr_dense.data, w_dlr, BSS.beta, eps=adapol_tol, statistics="Fermion", verbose=True)


if __name__ == '__main__':

    adapol_fit_test()