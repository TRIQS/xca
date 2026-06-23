
import numpy as np


from triqs.utility import mpi
from triqs.gfs import MeshDLRImTime


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
