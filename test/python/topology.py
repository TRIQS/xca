
import numpy as np

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_dlr, inverse, iOmega_n, make_gf_imtime


from triqs_xca.block_sparse_solver import BlockSparseSolver


def test_topology_parity_one_fermion(mu=0.1, beta=0.2, w_max=4.0, eps=1e-8, max_order=1):

    # NB! The order=2 sigma is zero for one fermion

    from triqs.operators import n

    Sd = BlockSparseSolver(
        H_loc=-mu * n('0', 0), beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 1]], 
        conserved_operators=[])

    Sd.Delta_tau['0'].data[:] = -0.5
    Sd.solve(max_order=max_order, spgf_max_order=1, maxiter=1, tol=1e-8, verbose=True, hyb_comp=False)

    Sbs = BlockSparseSolver(
        H_loc=-mu * n('0', 0), beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 1]])

    Sbs.Delta_tau['0'].data[:] = -0.5
    Sbs.solve(max_order=max_order, spgf_max_order=1, maxiter=1, tol=1e-8, verbose=True, hyb_comp=False)

    np.testing.assert_array_almost_equal(Sd.Sigma['0'].data[:, 0, 0], Sbs.Sigma['1'].data.flatten())
    np.testing.assert_array_almost_equal(Sd.Sigma['0'].data[:, 1, 1], Sbs.Sigma['0'].data.flatten())

    np.testing.assert_array_almost_equal(Sd.G_tau['0'].data, Sbs.G_tau['0'].data)


def test_topology_parity_two_fermions(mu=0.1 + 0.5, U=1.0, B=0.3, beta=2.0, w_max=4.0, eps=1e-10, max_order=2):

    from triqs.operators import n

    H_loc = \
        -mu * (n('0', 0) + n('0', 1)) + \
        B * (n('0', 0) - n('0', 1)) + \
        U * n('0', 0) * n('0', 1)
    
    Sd = BlockSparseSolver(
        H_loc=H_loc, beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 2]], 
        conserved_operators=[])

    Sd.Delta_tau['0'].data[:, 0, 0] = -0.5
    Sd.Delta_tau['0'].data[:, 1, 1] = -0.5

    print(Sd.atom_diag.energies)
    print(f'Sd.eta0 = {Sd.eta0}')
    rho0_d = Sd.many_body_density_matrix(Sd.G0)
    print(f'rho0_d =\n{rho0_d}')

    Sbs = BlockSparseSolver(
        H_loc=H_loc, beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 2]])

    Sbs.Delta_tau['0'].data[:, 0, 0] = -0.5
    Sbs.Delta_tau['0'].data[:, 1, 1] = -0.5
    
    print(Sbs.atom_diag.energies)
    print(f'Sbs.eta0 = {Sbs.eta0}')
    rho0_bs = Sbs.many_body_density_matrix(Sbs.G0)
    print(f'rho0_bs =\n{rho0_bs}')

    np.testing.assert_almost_equal(Sd.eta0, Sbs.eta0)

    sector_order = ['3', '1', '0', '2'] 

    for idx, sec in enumerate(sector_order):
        np.testing.assert_array_almost_equal(Sd.G0['0'].data[:, idx, idx], Sbs.G0[sec].data.flatten())
        np.testing.assert_array_almost_equal(Sd.G['0'].data[:, idx, idx], Sbs.G[sec].data.flatten())

    Sd.solve( max_order=max_order, spgf_max_order=1, maxiter=1, tol=1e-8, verbose=True, hyb_comp=True)
    Sbs.solve(max_order=max_order, spgf_max_order=1, maxiter=1, tol=1e-8, verbose=True, hyb_comp=True)

    for idx, sec in enumerate(sector_order):
        print(f'idx = {idx}, sec = {sec}')
        np.testing.assert_array_almost_equal(Sd.Sigma['0'].data[:, idx, idx], Sbs.Sigma[sec].data.flatten())

    np.testing.assert_array_almost_equal(Sd.G_tau['0'].data, Sbs.G_tau['0'].data)


if __name__ == "__main__":
    #test_topology_parity_one_fermion()
    test_topology_parity_two_fermions()
