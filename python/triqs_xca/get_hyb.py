import numpy as np
from scipy.integrate import quad

from .pycppdlr import build_dlr_rf
from .pycppdlr import ImTimeOps
from triqs.operators import c, n
from itertools import product
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori
from ac_pes import kernel, polefitting
from solver import Solver, eval_dlr_freq

def construct_impurity_Hamiltonian(norbI, U, J, mu):

    spin_names = ('up','do')
    orb_names = list(range(norbI))
    
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    
    KanMat1, KanMat2 = U_matrix_kanamori(norbI, U, J)
    H = h_int_kanamori(spin_names, norbI, KanMat1, KanMat2, J, off_diag = True)

    N_up = n('up',0) + n('up',1)
    N_do = n('do',0) + n('do',1)
    H -= mu *(N_up + N_do)
    return H,  fundamental_operators

def eval_semi_circ_tau(tau, beta, h, t):
    I = lambda x: -2 / np.pi / t**2 * kernel(np.array([tau]) / beta, beta * np.array([x]))[0, 0]
    g, res = quad(I, -t + h, t + h, weight='alg', wvar=(0.5, 0.5))
    return g

def construct_delta_iaa_semicircle(beta, r0 = 0.5):
    eval_semi_circ_tau_vec = np.vectorize(eval_semi_circ_tau)
    tau_l = np.real(np.linspace(0, beta, 500))  # Example time grid
    g = eval_semi_circ_tau_vec(tau_l, beta, h=0.0, t=2.0).reshape((len(tau_l), 1, 1))
    T = np.array([
        [1, r0, 0, 0],
        [r0, 1, 0, 0],
        [0, 0, 1, r0],
        [0, 0, r0, 1]])
    return T[None, ...] * g

def interp(g_xaa, ito, beta, tau_f):
    eval = lambda t : ito.coefs2vals(g_xaa, t / beta)
    return np.vectorize(eval, signature='()->(m,m)')(tau_f)

if __name__ == "__main__":
    eps = 1e-6
    beta = 8.0
    lamb = 10*beta
    eps_fitting = 1e-10
    U = 2.0
    J = 0.2
    dmu = -1.2
    mu = (3 * U - 5 * J) / 2.0 + dmu
    norbI = 2

    """
    delta_iaa = construct_delta_iaa_semicircle(beta)
    dlr_rf = build_dlr_rf(delta_iaa, beta, eps=eps)
    ito = ImTimeOps(lamb, dlr_rf)
    delta_xaa = ito.vals2coefs(delta_iaa)
    Deltaiw = eval_dlr_freq(delta_xaa, beta, eps=eps)

    nwmax = int(lamb) if lamb % 2 == 0 else int(lamb) + 1
    dlr_if = np.arange(-nwmax, nwmax + 1, 2) * np.pi / beta
    dlr_it = ito.get_itnodes()
    dlr_it_act = [t if t >= 0 else 1.0 + t for t in dlr_it]
    tau_f = np.linspace(0, beta, int(10 * lamb))
    Deltat = interp(delta_xaa, ito, beta, tau_f)
    Npmax = Deltaiw.shape[0] - 1

    weights, pol, error = polefitting(Deltaiw, 1.j * dlr_if, delta_iaa, dlr_it_act, Deltat, tau_f, beta, Np_max=Npmax, eps=eps_fitting, Hermitian=True)

    print("Fitted poles:", pol)
    print("Fitted weights:", weights)
    """
    H_loc, fundamental_operators = construct_impurity_Hamiltonian(norbI, U, J, mu)
    Impurity = Solver(beta, lamb, eps, H_loc, fundamental_operators)
    delta_iaa = construct_delta_iaa_semicircle(beta)
    Impurity.set_hybridization(delta_iaa, compress=True, delta_diff=1.0, fittingeps=eps_fitting, verbose=True, Hermitian=True, svd_trunc=False)
