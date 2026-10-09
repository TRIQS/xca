from itertools import product
import time
import numpy as np
from h5 import HDFArchive
from mpi4py import MPI as mpi

from triqs.operators import c, c_dag, n, Operator
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization


from scipy.integrate import quad

from triqs_xca.pycppdlr import build_dlr_rf
from triqs_xca.pycppdlr import ImTimeOps

from triqs_xca.solver import *
def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel
def H_soc_from_levi_cevita(lamb_soc):

    """ Constructs the Triqs operator for spin-orbit coupling
    in the cubic harmonic t2g basis.

    H_{soc} = \sum_{abc} \sum_{s_1, s_2}
        \epsilon_{abc} \sigma^{c}_{s_1, s_2} c^\dagger_{a s_1} c_{b s_2}

    where $a,b,c \in \{x, y, z\}$ and/or $\{yz, xz, xy\}$
    
    The orbital indices [0, 1, 2] are mapped to the cubic harmonics
    according to [0, 1, 2] -> [yz, xz, xy].

    Author: Hugo U. R. Strand """

    norb = 3
    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    #print(f'fundamental_operators = \n{fundamental_operators}')

    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_y = np.array([[0., -1.j], [1.j, 0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    sigma_vec = np.array([sigma_x, sigma_y, sigma_z])
    
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    H_soc = Operator()
    for i, j, k in product(range(norb), repeat=3):
        for s1, s2 in product(range(2), repeat=2):
            S1, S2 = spin_names[s1], spin_names[s2]
            H_soc += lamb_soc * 0.5j * \
                eijk[i, j, k] * sigma_vec[k, s1, s2] * c_dag(S1, i) * c(S2, j)

    return H_soc



def eval_semi_circ_tau(tau, beta, h, t):
    I = lambda x : -2 / np.pi / t**2 * \
        kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    return g

eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)

def construct_delta_iaa(tau_i, beta, T_diag, r, norb):
    delta_iaa = np.zeros((r, 2*norb, 2*norb), dtype=np.complex128)
    for idx, t_i in enumerate(T_diag):
        delta_iaa[:, idx, idx] = \
            t_i**2 * eval_semi_circ_tau(tau_i, beta, h=0.0, t=t_i)   
    return delta_iaa

def construct_impurity_Hamiltonian(norb, U, J, mu, delta_cf):

    spin_names = ('up','do')
    orb_names = list(range(norb))
    
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
    return H,  fundamental_operators
def run_calc(beta, lamb, eps, order,
             mu=3.9530058540332917,
             lamb_soc=0.,compress=True,
             write_h5=False, plot_flag=True,ntau=1000):
    max_order=order

    U, J, norb, delta_cf = 4.6, 0.8, 3,-1.0
    half_filling_shift = 0.5*(5*U - 10*J)
    mu += half_filling_shift
    t_diag = np.array([0.5, 0.5, 1.0])
    T_diag = np.concatenate([t_diag, t_diag])
    
    ppsc_tol, ppsc_maxiter = 1e-6, 1
    dmft_tol, dmft_maxiter = 1e-6, 100

    verbose = True

    #construct local Hamiltonians
    H,  fundamental_operators = construct_impurity_Hamiltonian(norb, U, J, mu, delta_cf)
    
    #itops tools
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    r = ito.rank()
    if is_root(): print(f'N_dlr = {r}')   
    
    
    g_iaa = np.zeros((r, 2*norb, 2*norb), dtype=np.complex128)

    delta_diff = np.inf 
    for dmft_iter in range(dmft_maxiter):
        if dmft_iter == 0:
            Impurity = Solver(beta, lamb, eps, H, fundamental_operators)
        else:
            G0_iaa = Impurity.G_iaa
            Impurity = Solver(beta, lamb, eps, H, fundamental_operators, G0_iaa = G0_iaa)
        if dmft_iter==0: delta_iaa = construct_delta_iaa(Impurity.fd.get_it_actual().real, beta, T_diag, r, norb) 
        Impurity.set_hybridization(delta_iaa = delta_iaa,compress=compress,delta_diff = delta_diff,fittingeps = 1e-7,verbose=True)
        Impurity.solve(max_order,  maxiter=ppsc_maxiter,tol = ppsc_tol, update_eta_exact = True , verbose=verbose)          
        g_iaa_old = g_iaa
        #calculate single-particle Green's functions diagrams
        g_iaa = Impurity.calc_spgf(max_order, verbose=verbose)
        # g_iaa = G_calc_loop(fd, G_iaa, max_order,delta_iaa.shape[1],verbose=True)
        #calculate new hybridization function
        delta_iaa = np.einsum('a,iab,b->iab', T_diag, g_iaa, T_diag)

        delta_diff = np.max(np.abs(g_iaa - g_iaa_old))

        g_xaa = ito.vals2coefs(g_iaa)
        rho_aa = -ito.coefs2eval(g_xaa, 1.0)
        N = np.sum(np.diag(rho_aa)).real         
        if is_root():  
            print(
                f'DMFT: iter {dmft_iter:3d}, ddelta {delta_diff:2.2E}' + \
                ' - '  + f' - N {N:8.8E}')

        if delta_diff < dmft_tol: break



    N = np.sum(np.diag(rho_aa)).real
    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)


    tau_f = np.linspace(0, beta, num=ntau)

    g_xaa, delta_xaa, G_xaa, Sigma_xaa = ito.vals2coefs(g_iaa), ito.vals2coefs(delta_iaa),ito.vals2coefs(Impurity.G_iaa), ito.vals2coefs(Impurity.Sigma_iaa)
    g_faa,delta_faa, G_faa, Sigma_faa = interp(g_xaa, tau_f), interp(delta_xaa, tau_f), interp(G_xaa, tau_f),interp(Sigma_xaa, tau_f)
    class Dummy():
        def __init__(self):
            pass

    d = Dummy()
    d.N = N
    d.g_faa,d.delta_faa, d.G_faa, d.Sigma_faa = g_faa,delta_faa, G_faa, Sigma_faa

    return d

            
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

    order = 1
    
    parms = [
        ( 5., 100., 1e-8,order),
        # ( 5., 1000., 1e-8),
        ]

    N_target = 4.0

    mu_nca = 3.977224604033292
    mu_oca = 3.965505854033292
    mu_tca = 3.9655058540332924
    mulist = [mu_nca, mu_oca, mu_tca]
    mu = mulist[order-1]
    #mu = mu_nca
    #mu = mu_oca
    # mu = mu_tca

    mu0 = mu - 0.05
    mu1 = mu + 0.05

    lamb_socs = [0.2]
    for beta, lamb, eps,order in parms:

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
                             mu=mu,compress=True,
                             write_h5=write_h5, plot_flag = False)
                return p.N

            #mu = fix_N_calc(N_target, mu0, mu1, N_func)

            if is_root():
                print('='*72)
                print('='*72)
                print('--> Final')
                print(f'mu = {mu}')
                print('='*72)
                print('='*72)

            # p = N_func(mu, write_h5=True)
            compress=True
            result = run_calc(beta, lamb, eps, order,
                             lamb_soc=lamb_soc,
                             mu=mu,compress=compress,
                             write_h5=False, plot_flag = False)
            np.save("dmft_order="+str(order)+"_compress="+str(compress)+".npy",result.g_faa)
            if is_root():
                breakpoint()
            

            
        

