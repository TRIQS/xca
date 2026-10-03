"""
Test asymptotic convergence of the single particle Green's function
and the density-density correlation function for a non-interacting 
n-level AIM with discrete bath.

The convergence is tested for both the dense and block-sparse diagram evaluators.

Author: Hugo U. R. Strand (2026) 
"""

import numpy as np


from triqs.utility import mpi
from triqs.gfs import MeshDLRImTime


from triqs_xca import Solver


class Dummy:
    def __init__(self): pass


def plot_comparison(m_dlr, ed_solver, xca_solver, max_order=4, t=-0.5):

    """ Compare XCA solution to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    ed = ed_solver(m_dlr, t=t)

    oxs = []
    for order in range(1, max_order+1):
         print(f'Computing XCA solution for order {order}...')
         ox = xca_solver(m_dlr, t=t, sigma_order=order, verbose=True)
         oxs.append(ox)

    if mpi.is_master_node():

        for ox in oxs:
            G_err = np.max(np.abs((ox.G_tau - ed.G_tau).data))
            Chi_err = np.max(np.abs((ox.Chi_tau - ed.Chi_tau).data))
            print(f'O{ox.sigma_order} error: G={G_err:.3e}, Chi={Chi_err:.3e}')


    if mpi.is_master_node():

        from triqs.plot.mpl_interface import oplot, plt
        plt.figure(figsize=(6, 8))
        subp = [3, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.G_tau.real, marker='+', label=f'O{ox.sigma_order} sc')
        oplot(ed.G_tau.real, marker='x', lw=4., alpha=0.5, label='ED')
        plt.ylabel(r'$g(\tau)$')
        plt.ylim(top=0.)

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.Chi_tau.real, marker='+', label=f'O{ox.spgf_order} sc')
        oplot(ed.Chi_tau.real, marker='x', lw=4., alpha=0.5, label='ED')
        plt.ylabel(r'$\chi_{nn}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.Chi_tau.real - ed.Chi_tau.real, marker='+', label=f'O{ox.spgf_order} sc')
        plt.ylabel(r'Err $\chi_{nn}(\tau)$')

        plt.tight_layout()
        plt.show()


def test_convergence_rate(m_dlr, ed_solver, xca_solver, label='dimer', max_order=5, do_test=False, verbose=True):

    """ Test convergence rate of the dynamic interaction expansion 
    by comparing to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    G_errss = []
    Chi_errss = []
    orders = list(range(1, max_order + 1))
    for order in orders:
        t2s = np.logspace(-1.5, -1.0, 2)
        #t2s = np.logspace(-1.5, 0.5, 4)
        G_errs = np.zeros_like(t2s)
        Chi_errs = np.zeros_like(t2s)

        for i, t2 in enumerate(t2s):

            t = -np.sqrt(t2)
            ed = ed_solver(m_dlr, t=t)
            xca = xca_solver(m_dlr, t=t, sigma_order=order, verbose=True)

            G_errs[i] = np.max(np.abs((xca.G_tau - ed.G_tau).data))
            Chi_errs[i] = np.max(np.abs((xca.Chi_tau - ed.Chi_tau).data))
            if mpi.is_master_node():
                print(f'order={order}, t={t:.3f}, error: G={G_errs[i]:.3e}, Chi={Chi_errs[i]:.3e}')

        G_errss.append(G_errs)
        Chi_errss.append(Chi_errs)

    if mpi.is_master_node():
        # Compute convergence rates
        G_rates = []
        Chi_rates = []
        for order, G_errs, Chi_errs in zip(orders, G_errss, Chi_errss):
            G_rate = (np.log(G_errs[:-1] / G_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            Chi_rate = (np.log(Chi_errs[:-1] / Chi_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            G_rates.append(G_rate)
            Chi_rates.append(Chi_rate)
            #print(f'Label: {label}')
            #print(f'Order {order} convergence rates: G={G_rate:.2f}, Chi={Chi_rate:.2f}')

    if verbose and mpi.is_master_node():
        # Plot errors
        import matplotlib.pyplot as plt

        for i, (order, G_errs, Chi_errs) in enumerate(zip(orders, G_errss, Chi_errss)):

            x = [t2s[0], t2s[0] + 0.2 * (t2s[-1] - t2s[0])]
            y = (x / x[0])**order

            subp = [1, 2, 1]
            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, G_errs, 'o-', label=f'O{order}', alpha=0.75)
            plt.plot(x, G_errs[0] * y , 'k--', lw=0.5)
            plt.xlabel('$t^2$')
            plt.ylabel(r'Error: $\max_i |g(\tau_i) - g^{\text{ED}}(\tau_i)|$')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, Chi_errs, 'o-', label=f'O{order}', alpha=0.75)
            plt.plot(x, Chi_errs[0] * y, 'k--', lw=0.5)
            plt.xlabel('$t^2$')
            plt.ylabel(r'Error: $\max_i |\chi_{nn}(\tau_i) - \chi_{nn}^{\text{ED}}(\tau_i)|$')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

        plt.tight_layout()
        plt.savefig(f'figure_{label}_convergence.pdf')
        plt.show()

    if mpi.is_master_node():
        # Check convergence rates
        for order, G_rate, Chi_rate in zip(orders, G_rates, Chi_rates):
            if do_test:
                print(f'Label: {label} order {order} convergence rates: G={G_rate:.2f}, Chi={Chi_rate:.2f}')
                assert( G_rate > order - 0.3 ), f'Expected convergence rate of {order} for order {order}, but got {G_rate}.'
                assert( Chi_rate > order - 0.3 ), f'Expected convergence rate of {order} for order {order}, but got {Chi_rate}.'


def analytic_solution(mesh_tau, t=-1.0, e0=0.1):

    from triqs.gfs import Gf, make_gf_dlr_imfreq, inverse, iOmega_n, SemiCircular, make_gf_dlr_imtime, make_gf_dlr
    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    G_iw = make_gf_dlr_imfreq(G_tau)
    Delta_iw = G_iw.copy()

    Delta_iw << t**2 * inverse(iOmega_n)

    G_iw << inverse(iOmega_n - e0 - Delta_iw)
    G_tau = make_gf_dlr_imtime(G_iw)

    Chi_tau = Gf(mesh=mesh_tau, target_shape=[])

    G_dlr = make_gf_dlr(G_tau)
    beta = mesh_tau.beta
    n_exp = -G_dlr(beta)
    for tau in mesh_tau:
        Chi_tau[tau] = -G_tau[tau] * G_dlr(beta-tau) - n_exp**2

    d = Dummy()
    d.G_tau = G_tau
    d.Chi_tau = Chi_tau
    return d


def xca_n_level_solution(
        mesh_tau, N=2,t=-1.0, e0=0.1, sigma_order=1, 
        spgf_order=None, verbose=False, conserved_operators=[]):

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    from triqs.operators import n, c, c_dag

    H_loc = e0 * sum(n('0', i) for i in range(N))

    S = Solver(
        H_loc=H_loc, gf_struct=[['0', N]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=conserved_operators,
        )

    for idx in range(N):
        S.Delta_tau['0'].data[:, idx, idx] = -0.5 * t**2

    S.solve(max_order=sigma_order, spgf_max_order=spgf_order, 
            tol=1e-12, maxiter=20, verbose=True)

    S.G_tau_ref = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[c('0', 0)], ops_0=[c_dag('0', 0)])

    np.testing.assert_array_almost_equal(S.G_tau['0'][0, 0].data, S.G_tau_ref[0, 0].data)

    S.Chi_tau = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    d = Dummy()
    d.S = S
    d.sigma_order = sigma_order
    d.spgf_order = spgf_order
    d.G_tau = S.G_tau['0'][0, 0].copy()
    d.Chi_tau = S.Chi_tau[0, 0].copy()
    return d


if __name__ == "__main__":

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-12, w_max=8.0, symmetrize=False)

    #plot_comparison(m_dlr, analytic_two_level_solution, xca_two_level_solution_dense, max_order=3)

    for N in range(1, 2+1):
        for conserved_operators in [[], 'automatic']:

            xca_nlvl = lambda mesh_tau, t, sigma_order, verbose : \
                xca_n_level_solution(mesh_tau, N=N, t=t, sigma_order=sigma_order, conserved_operators=conserved_operators, verbose=verbose)

            label = f'{N}_level'

            if conserved_operators == 'automatic':
                label += '_block_sparse'
            else:
                label += '_dense'

            test_convergence_rate(m_dlr, analytic_solution, xca_nlvl, 
                                  label=label, max_order=3, do_test=True, verbose=False)
