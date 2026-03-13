import numpy as np


def from_blockgf_to_dense(G):

    for b, g in G:
        assert( len(g.target_shape) == 2)
        assert( g.target_shape[0] == g.target_shape[1] )

    norb = sum([ g.target_shape[0] for b, g in G ])
    
    from triqs.gf import Gf
    G_dense = Gf(mesh=G.mesh, target_shape=[norb]*2)
    
    sidx = 0
    for b, g in G:
        size = g.target_shape[0]
        G_dense.data[:, sidx:sidx+size, sidx:sidx+size] = g.data
        sidx += size

    return G_dense


def adapol_fit_test(do_xca_calc=True):

    beta = 3.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-12
    w_max = 10.0 * beta

    order = 1
    maxiter = 100
    ppsc_tol = 1e-14
    adapol_tol = 1e-10

    if do_xca_calc:

        gf_struct = [['0', 1]]

        from triqs.operators import n
        H = -mu * n('0', 0)
        
        from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime, make_gf_dlr

        mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
        Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

        Delta_w << 0.5 * inverse(iOmega_n - e1)
        Delta_tau = make_gf_dlr_imtime(Delta_w)

        # -- Block sparse solver
        from triqs_xca.block_sparse_solver import BlockSparseSolver
        BSS = BlockSparseSolver(H, beta, w_max, eps, gf_struct)
        BSS.Delta_tau['0'] << Delta_tau
        BSS.solve(max_order=order, tol=ppsc_tol, maxiter=maxiter)
    
        g_BSS = BSS.G_tau

        BSS.Delta_tau << BSS.G_tau # Bethe self-consistency

        BSS.fit_hybridization(tol=adapol_tol)

        if False:
            from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
            oplot(BSS.Delta_tau)
            plt.show()
            exit()

        Delta_dlr = make_gf_dlr(BSS.Delta_tau)

        from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense
        Delta_dlr_dense = from_blockgf_to_dense(Delta_dlr)

        w_dlr = np.array([ float(x) for x in Delta_dlr.mesh ])
        Delta_dlr_data = Delta_dlr_dense.data[:, 0, 0].real

    else:
        Delta_dlr_data = np.array([-2.87059648e-03,  8.03697280e-03, -7.54738032e-03,  5.63683166e-03,
       -1.65833193e-02,  2.65791603e-02, -2.20980377e-02,  2.10371505e-02,
       -2.40439486e-02,  2.84997872e-02, -2.49267923e-02,  1.65782606e-02,
       -2.78233831e-02,  5.86472915e-02, -1.46491003e-01,  6.52025753e-01,
        3.17576895e-01, -1.02506149e-02,  2.63118605e-01, -2.26965298e-01,
        1.40062175e-01, -6.32622719e-02,  5.89104167e-02, -2.72432219e-02,
        5.05731934e-03, -3.90840782e-03,  5.09995648e-03, -4.68438177e-03,
        2.91969618e-03, -1.67744675e-03,  1.71064218e-03, -1.70190617e-03,
        5.81097236e-04])
        w_dlr = np.array([-8.99518258e+01, -8.76796367e+01, -8.44163957e+01, -7.47323880e+01,
       -6.31104678e+01, -5.75485045e+01, -5.05836043e+01, -4.11676404e+01,
       -3.44857852e+01, -2.74998349e+01, -2.30970360e+01, -1.65071074e+01,
       -1.05520495e+01, -6.87495872e+00, -3.76672575e+00, -1.64577247e+00,
        1.50544460e-03,  2.15536158e+00,  5.67904140e+00,  7.88880847e+00,
        1.02919101e+01,  1.57776169e+01,  1.93628739e+01,  2.27161656e+01,
        3.15552339e+01,  4.22081978e+01,  5.05836043e+01,  5.75485045e+01,
        6.60284296e+01,  7.47323880e+01,  8.44163957e+01,  8.76796367e+01,
        8.99518258e+01])


    print('Delta_dlr_data =\n', repr(Delta_dlr_data))
    print('w_dlr =\n', repr(w_dlr))

    print('='*72)
    print('='*72)
    print('--> Calling adapol.fit_utils_dlr.polefitting_dlr')
    print('='*72)
    print('='*72)

    from adapol.fit_utils_dlr import polefitting_dlr

    pole_weights, poles, fit_error = polefitting_dlr(
        Delta_dlr_data.reshape((len(w_dlr), 1, 1)), w_dlr, beta, eps=adapol_tol, statistics="Fermion", verbose=True)


if __name__ == '__main__':

    adapol_fit_test()