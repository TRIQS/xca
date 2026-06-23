
import numpy as np


from triqs.utility import mpi
from triqs.gfs import MeshDLRImTime


class Dummy:
    def __init__(self): pass


def plot_comparison(m_dlr, ed_solver, xca_solver, max_order=4):

    """ Compare XCA solution to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    t = -0.5

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


def test_convergence_rate(m_dlr, ed_solver, xca_solver, label='dimer', max_order=5):

    """ Test convergence rate of the dynamic interaction expansion 
    by comparing to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    G_errss = []
    Chi_errss = []
    orders = list(range(1, max_order + 1))
    for order in orders:
        #t2s = np.logspace(-1.5, -0.5, 3)
        t2s = np.logspace(-1.5, 0.5, 4)
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
        for order, G_errs, Chi_errs in zip(orders, G_errss, Chi_errss):
            G_rate = (np.log(G_errs[:-1] / G_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            Chi_rate = (np.log(Chi_errs[:-1] / Chi_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            print(f'Order {order} convergence rates: G={G_rate:.2f}, Chi={Chi_rate:.2f}')

    if mpi.is_master_node():
        # Plot errors
        import matplotlib.pyplot as plt

        for i, (order, G_errs, Chi_errs) in enumerate(zip(orders, G_errss, Chi_errss)):
            subp = [1, 2, 1]
            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, G_errs, 'o-', label=f'O{order} G')
            plt.xlabel('$t^2$')
            plt.ylabel(r'$g(\tau)$ Error')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, Chi_errs, 's-', label=f'O{order} Chi')
            plt.xlabel('$t^2$')
            plt.ylabel(r'$\chi_{nn}(\tau)$ Error')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

        plt.tight_layout()
        plt.savefig(f'figure_{label}_convergence.pdf')
        plt.show()