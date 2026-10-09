
from itertools import product

import numpy as np

from h5 import HDFArchive
from mpi4py import MPI as mpi

from triqs.operators import c, c_dag, n, Operator
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori

from t2g_soc import H_soc_from_levi_cevita

from threebanddmft import kernel
from scipy.integrate import quad

from triqs_xca.solver import Solver


def is_root():
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    return rank == 0


def eval_semi_circ_tau(tau, beta, h, t):
    I = lambda x : -2 / np.pi / t**2 * \
        kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    return g

eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)


def run_calc(beta, lamb, eps, order,
             mu=3.9530058540332917,
             lamb_soc=0.,
             mix=1.0,
             write_h5=False, plot_flag=True, delta_iaa=None, G_iaa=None, timer=None):

    U = 4.6
    J = 0.8
    norb = 3
    
    half_filling_shift = 0.5*(5*U - 10*J)
    mu += half_filling_shift
    t_diag = np.array([0.5, 0.5, 1.0])
    T_diag = np.concatenate([t_diag, t_diag])
    delta_cf = -1.0
    
    ppsc_tol = 1e-6
    ppsc_maxiter = 1
    dmft_tol = 1e-6
    dmft_maxiter = 100 # self-cons settings

    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]

    KanMat1, KanMat2 = U_matrix_kanamori(norb, U, J)
    H = h_int_kanamori(spin_names, norb, KanMat1, KanMat2, J, off_diag=True)

    N_0 = n('up', 0) + n('do', 0)
    N_2 = n('up', 2) + n('do', 2)
    N_up = sum([ n('up', idx) for idx in range(norb) ])
    N_do = sum([ n('do', idx) for idx in range(norb) ])

    H -= mu *(N_up + N_do)
    H += delta_cf * N_2 # using that orbital 2 is xy

    H += H_soc_from_levi_cevita(lamb_soc)
    
    # Setup impurity problem    
    
    S = Solver(beta, lamb, eps, H, fundamental_operators, timer=timer)
    ito = S.ito
    
    tau_i = S.tau_i

    if delta_iaa is None:
        delta_iaa = np.zeros((ito.rank(), 2*norb, 2*norb), dtype=S.H_mat.dtype)
        for idx, t_i in enumerate(T_diag):
            delta_iaa[:, idx, idx] = \
                t_i**2 * eval_semi_circ_tau(tau_i, beta, h=0.0, t=t_i)

    g_iaa = None
    
    delta_diff = 1.0

    for dmft_iter in range(dmft_maxiter):

        S.set_hybridization(
            delta_iaa, compress=True,
            delta_diff=delta_diff, fittingeps=10*eps, Hermitian=True, verbose=True)
            
        S.solve(order, tol=ppsc_tol, maxiter=ppsc_maxiter, mix=mix, update_eta_exact=True, verbose=True, G0_iaa=G_iaa)

        G_iaa = S.G_iaa
        Sigma_iaa = S.Sigma_iaa

        g_iaa_old = g_iaa
        g_iaa = S.calc_spgf(order)
                
        delta_iaa_old = delta_iaa
        delta_iaa = np.einsum('a,iab,b->iab', T_diag, g_iaa, T_diag)

        delta_diff = np.max(np.abs(delta_iaa - delta_iaa_old))
        
        rho_GG = S.get_many_body_density_matrix()
        rho_aa = S.get_single_particle_density_matrix()
        N = S.get_density()
                 
        if is_root():
            print(
                f'DMFT: iter {dmft_iter:3d}, ddelta {delta_diff:2.2E}' + \
                #' - ' + f'PPSC: iter {ppsc_iter:3d}, dppgf {ppsc_diff:2.2E}' + \
                f' - N {N:8.8E}')

        if delta_diff < dmft_tol: break

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)

    tau_f = np.linspace(0, beta, num=100)

    g_xaa = ito.vals2coefs(g_iaa)
    g_faa = interp(g_xaa, tau_f)

    delta_xaa = ito.vals2coefs(delta_iaa)
    delta_faa = interp(delta_xaa, tau_f)

    G_xaa = ito.vals2coefs(G_iaa)
    G_faa = interp(G_xaa, tau_f)

    Sigma_xaa = ito.vals2coefs(Sigma_iaa)
    Sigma_faa = interp(Sigma_xaa, tau_f)

    if is_root() and write_h5:
        filename = f'data_cro_fastdiag_{order}_beta_{beta}_soc_{lamb_soc}.h5'
        print(f'--> Writing: {filename}')
        with HDFArchive(filename, 'w') as A:

            A['g_iaa'] = g_iaa
            A['delta_iaa'] = delta_iaa
            A['tau_i'] = tau_i
            
            A['g_faa'] = g_faa
            A['delta_faa'] = delta_faa
            A['tau_f'] = tau_f

            A['G_faa'] = G_faa
            A['Sigma_faa'] = Sigma_faa

            A['rho'] = rho_aa
            
            A['beta'] = beta
            A['order'] = order

            A['lamb'] = lamb
            A['eps'] = eps

            A['H'] = H
            
            A['U'] = U
            A['J'] = J
            A['mu'] = mu
            A['t_diag'] = t_diag
            A['delta_cf'] = delta_cf
            A['lamb_soc'] = lamb_soc
            A['N'] = N

    if is_root() and plot_flag:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        subp = [6, 6, 1]

        nspinorb = 6
        rng = [0, 1, 5, 3, 4, 2]
        lables = [
            r'$yz \uparrow$',
            r'$xz \uparrow$',
            r'$xy \uparrow$',
            r'$yz \downarrow$',
            r'$xz \downarrow$',
            r'$xy \downarrow$',
            ]
        for a, b in product(rng, repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            la, lb = lables[a], lables[b]
            plt.title(la + ', ' + lb, fontsize=7)
            for flt in [np.real, np.imag]:
                l = plt.plot(
                    tau_i, flt(g_iaa[:, a, b]),
                    '.', label=f'g {a},{b}', alpha=0.5)
                color = l[0].get_color()
                plt.plot(
                    tau_f, flt(g_faa[:, a, b]),
                    '-', color=color, alpha=0.75)
            #plt.xlabel(r'$\tau$')
            if a != b: plt.ylim([-0.1, 0.1])
            else: plt.ylim([-1, 0.05])

        plt.tight_layout()
        plt.savefig(
            f'figure_fastdiag_cro_order_{order}_lamb_soc_{lamb_soc}.pdf')
        plt.show()

    return delta_iaa, G_iaa, S

            
def fix_N_calc(N_target, mu0, mu1, func):

    def target_function(mu):
        N = func(mu)
        print('='*72)
        print(f'mu = {mu}, N = {N}')
        print('='*72)
        return N_target - N

    from scipy.optimize import root_scalar

    sol = root_scalar(
        target_function, bracket=[mu0, mu1],
        x0=mu0, x1=mu1, method='bisect', xtol=1e-3)
    
    return sol.root


if __name__ == '__main__':

    order = 2
    
    parms = [
        #( 5., 200., 1e-9),
        ( 5., 100., 1e-8),
        #( 5., 1000., 1e-12),
        ]

    N_target = 4.0

    mu_nca = 3.977224604033292
    mu_oca = 3.965505854033292
    mu_tca = 3.9655058540332924

    #mu = mu_nca
    #mu = mu_oca
    mu = mu_tca # NB! Use mu_tca for all orders to compare with prev code.

    mu0 = mu - 0.05
    mu1 = mu + 0.05

    lamb_socs = [0.2]
    
    for beta, lamb, eps in parms:

        for lamb_soc in lamb_socs:

            if is_root():
                print('='*72)
                print('='*72)
                print(f'lamb_soc = {lamb_soc}')
                print('='*72)
                print('='*72)
            
            def N_func(mu, write_h5=False):
                p = run_calc(beta, lamb, eps, order,
                             lamb_soc=lamb_soc,
                             mu=mu,
                             write_h5=write_h5)
                return p.N

            #mu = fix_N_calc(N_target, mu0, mu1, N_func)

            if is_root():
                print('='*72)
                print('='*72)
                print('--> Final')
                print(f'mu = {mu}')
                print('='*72)
                print('='*72)

            #p = N_func(mu, write_h5=True)
            
            opts = dict(lamb_soc=lamb_soc, mu=mu, write_h5=True, plot_flag=False, G_iaa=None, delta_iaa=None)

            opts['delta_iaa'], opts['G_iaa'], S = run_calc(beta, lamb, eps, 1, **opts)
            opts['delta_iaa'], opts['G_iaa'], S = run_calc(beta, lamb, eps, 2, timer=S.timer, **opts)
            #opts['delta_iaa'], opts['G_iaa'], S = run_calc(beta, lamb, eps, 3, timer=S.timer, **opts)
        
