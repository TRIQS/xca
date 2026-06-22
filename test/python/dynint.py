""" Test dynamic interaction expansion for a single fermionic level coupled to a bosonic mode.

Author: Hugo U. R. Strand, 2026 """

import numpy as np

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_dlr, inverse, iOmega_n, make_gf_imtime


from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators


from triqs_xca.block_sparse_solver import BlockSparseSolver


def get_ed_ref(eps0, g, omega0, mesh_f_tau, mesh_b_tau, Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=1, Nb=1, Nb_max=Nb_max)
    
    c, b = ops.c_dag[0].getH(), ops.b_dag[0].getH()
    nf, nb = c.getH() * c, b.getH() * b

    H = eps0 * nf + g*(b + b.getH())*nf + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_f_tau.beta)

    tau_f = np.array([float(t) for t in mesh_f_tau])
    g_tau = Gf(mesh=mesh_f_tau, target_shape=[1, 1])
    g_tau.data[:, 0, 0] = ed.get_tau_greens_function_component(tau_f, c, c.getH())

    tau_b = np.array([float(t) for t in mesh_b_tau])
    chi_tau = Gf(mesh=mesh_b_tau, target_shape=[1, 1])
    chi_tau.data[:, 0, 0] = -ed.get_tau_greens_function_component(tau_b, nf, nf)
    
    return g_tau, chi_tau


def test_dynint_one_fermion(
        beta=2.1, eps0=-0.1, g=0.1, omega0=1., w_max=2.0, eps=1e-12, 
        order=1, verbose=False):
    
    """" Solve AIM with single fermionic level coupled to a bosonic mode 
    with linear coupling and retarded interaction given by the bosonic propagator. 
    Compare to ED reference solution. 
    
    Note that the retarded interaction does not contribute to the 
    single particle Green's function diagrams (they are zero for all orders).

    Thus, this only tests the pseudo particle self-energy diagrams.
    """

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    from triqs.operators import n

    S = BlockSparseSolver(
        H_loc=(eps0 - mu) * n('0', 0), 
        beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 1]], conserved_operators=[])
    
    S.Delta_tau['0'].data[:] = 0.

    f_mesh = S.mesh_tau
    b_mesh = MeshDLRImTime(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max)

    D0_tau = Gf(mesh=b_mesh, target_shape=[1, 1])
    D0_iw = make_gf_dlr_imfreq(D0_tau)
    D0_iw << -2 * g**2 * omega0 * inverse(omega0**2 - iOmega_n*iOmega_n)

    D0_iw << 0.5 * D0_iw # FIXME! Compensate for double number of Sigma diagrams for retarded interactions

    D0_tau << make_gf_dlr_imtime(D0_iw)
    D0_c = make_gf_dlr(D0_iw)

    r = S.Delta_tau['0'].data.shape[0]

    dynint_ops = [n('0', 0)]
    dynint_coeffs = np.zeros([r, 1, 1], dtype=complex)

    dynint_coeffs[:] = D0_c.data

    S.set_dynamic_interactions(dynint_ops, dynint_coeffs)
    S.solve(max_order=order, spgf_max_order=1, maxiter=8, tol=1e-8, verbose=True, hyb_comp=False)

    g_tau_ed_0, chi_tau_ed_0 = get_ed_ref(eps0 - mu, 0.0, omega0, f_mesh, b_mesh, Nb_max=10)
    g_tau_ed, chi_tau_ed = get_ed_ref(eps0 - mu, g, omega0, f_mesh, b_mesh, Nb_max=10)

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt

        plt.figure(figsize=(6, 6))
        subp = [1, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_imtime(S.G_tau, n_tau=100).real, '-', label='xca')
        oplot(make_gf_imtime(g_tau_ed, n_tau=100).real, ':', label='ed')
        oplot(make_gf_imtime(g_tau_ed_0, n_tau=100).real, ':', label='ed (g=0)')
        plt.show()

    error = np.max(np.abs(S.G_tau['0'].data - g_tau_ed.data))
    return error


def test_convergence_rate(verbose=False):

    """ Test convergence rate of the dynamic interaction expansion 
    by comparing to ED reference solution for a single fermionic level 
    coupled to a bosonic mode.
     
    At order = 1 we expect a convergence rate of 2 (error ~ g^4), and 
    at order = 2 we expect a convergence rate of 3 (error ~ g^6)."""

    errss = []
    orders = [1, 2]
    for order in orders:
        g2s = np.logspace(-1.5, -0.5, 3)
        errs = np.zeros_like(g2s)
        for i, g2 in enumerate(g2s):
            g = np.sqrt(g2)
            errs[i] = test_dynint_one_fermion(g=g, order=order)
        errss.append(errs)

    if mpi.is_master_node():
        # Compute convergence rates
        rates = []
        for order, errs in zip(orders, errss):
            rate = (np.log(errs[:-1] / errs[1:]) / np.log(g2s[:-1] / g2s[1:]))[0]
            rates.append(rate)
            print(f'Order {order} convergence rates: {rate}')

    if verbose and mpi.is_master_node():
        import matplotlib.pyplot as plt
        for i, (order, errs) in enumerate(zip(orders, errss)):
            plt.loglog(g2s, errs, 'o-', label=f'O{order}')
        plt.xlabel('$g^2$')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    if mpi.is_master_node():
        # Test convergence rates
        for order, rate in zip(orders, rates):
            diff = np.abs(rate - (order + 1))
            assert( diff < 0.2 ), f'Expected convergence rate of {order+1} for order {order}, but got {rate}, diff {diff}'


if __name__ == '__main__':
    test_convergence_rate(verbose=False)
