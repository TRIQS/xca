################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
#
# triqs_soehyb is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

""" Test basic functionality of the Python module wrapper of cppdlr::imtime_ops. """


import numpy as np


from mpi4py import MPI


import triqs_soehyb.pycppdlr as d


def kernel(tau, omega):

    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel


def free_greens_function_tau(H_aa, beta, tau_l):

    E, U = np.linalg.eigh(H_aa)
    g_lE = -kernel(tau_l/beta, E*beta)
    g_laa = np.einsum('lE,aE,Eb->lab', g_lE, U, U.T.conj())

    return g_laa
    

def test_pycppdlr(verbose=True):

    if verbose: print(dir(d))

    beta = 13.0
    lamb = 200.0
    eps = 1e-8

    dlr_rf = d.build_dlr_rf(lamb, eps)

    itops = d.ImTimeOps(lamb, dlr_rf)
    if verbose: print(dir(itops))

    assert( len(dlr_rf) == itops.rank() )
    assert( lamb == getattr(itops, 'lambda')() )

    np.testing.assert_array_almost_equal(dlr_rf, itops.get_rfnodes() )

    t_i = itops.get_itnodes()
    tau_i = beta * (t_i + (t_i < 0) * 1)

    if verbose:
        print(f't_i = {t_i}')
        print(f'tau_i = {tau_i}')

    cf2it = itops.get_cf2it()
    it2cf_lu = itops.get_it2cf_lu()
    it2cf_piv = itops.get_it2cf_piv()

    # -- Simple transform and interpolation test

    H_aa = np.array([[1.0]], dtype=complex)

    g_iaa = free_greens_function_tau(H_aa, beta, tau_i)
    g_xaa = itops.vals2coefs(g_iaa)

    g_iaa_refl = itops.reflect(g_iaa)
    g_iaa_refl_ref = free_greens_function_tau(-H_aa, beta, tau_i)

    tau_f = np.linspace(0, beta, num=100)
    g_faa = free_greens_function_tau(H_aa, beta, tau_f)
    g_faa_interp = np.array([ itops.coefs2eval(g_xaa, tau/beta) for tau in tau_f ])

    # -- Test convolution

    e1, e2 = 3., 3.3

    g1_iaa = free_greens_function_tau(np.array([[e1]]), beta, tau_i)
    g2_iaa = free_greens_function_tau(np.array([[e2]]), beta, tau_i)

    g1_xaa = itops.vals2coefs(g1_iaa)
    g2_xaa = itops.vals2coefs(g2_iaa)
    
    c = 1./(e1 - e2)
    gg_iaa_anal = c * g1_iaa - c * g2_iaa

    gg_iaa = itops.convolve(beta, "Fermion", g1_xaa, g2_xaa)
        
    if verbose:
        import matplotlib.pyplot as plt
        subp = [3, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(tau_i, g_iaa.flatten().real, 'o-')
        plt.plot(tau_f, g_faa.flatten().real, 'x')
        plt.plot(tau_f, g_faa_interp.flatten().real, '+')

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(tau_f, np.abs(g_faa - g_faa_interp).flatten().real, '.-')
        plt.semilogy([], [])

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(tau_i, g_iaa_refl.flatten().real, '.-')
        plt.plot(tau_i, g_iaa_refl_ref.flatten().real, 'x-')

        plt.tight_layout()
        plt.show()


    np.testing.assert_array_almost_equal(g_iaa_refl, g_iaa_refl_ref)

    np.testing.assert_array_almost_equal(g_faa, g_faa_interp)
    assert( np.max(np.abs(g_faa - g_faa_interp)) < eps )

    np.testing.assert_array_almost_equal(gg_iaa, gg_iaa_anal)    
    assert( np.max(np.abs(gg_iaa - gg_iaa_anal)) < eps ) 
    

if __name__ == "__main__":

    test_pycppdlr(verbose=False)
