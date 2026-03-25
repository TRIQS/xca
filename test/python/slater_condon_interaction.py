
import numpy as np

from triqs.operators.util.U_matrix import U_matrix_slater
from triqs.operators.util.hamiltonians import h_int_slater

from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori


from triqs_xca.block_sparse_solver import is_root


beta = 1.0
l = 2 # d-orbitals angular momentum quantum number

order = 1 # pseudo-particle hybridization expansion order

# Slater-Condon interaction parameters for d-orbitals
F0 = 3.0
F2 = 0.5
F4 = 0.3

mu = (45*F0 - 70/49*F2 - 630/441*F4) / 10. # For half-filling of the d-shell

eps = 1e-3
#w_max = 75.0 * F0 # 10 electrons with energy scale 75 * F0
w_max = 40.

ppsc_tol = 1e-2
ppsc_maxiter = 10

# --

n_orb = 2*l + 1
spin_names = ['up', 'do']
gf_struct = [ [f'{spin}', n_orb] for spin in spin_names ]

U_matrix = U_matrix_slater(l, radial_integrals=[F0, F2, F4])
H = h_int_slater(spin_names, n_orb, U_matrix, off_diag=True)

from triqs.operators import n
N_tot = sum( n(spin, oidx) for spin in spin_names for oidx in range(n_orb) )
H += - mu * N_tot

from triqs.gf import SemiCircular
from triqs.gf import MeshDLRImFreq, Gf, make_gf_dlr_imtime, iOmega_n, inverse

mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
Delta_w = Gf(mesh=mesh_w, target_shape=[n_orb]*2)

#Delta_w << inverse(iOmega_n)
Delta_w << SemiCircular(half_bandwidth=1.0)
Delta_tau = make_gf_dlr_imtime(Delta_w)

from triqs_xca.block_sparse_solver import BlockSparseSolver

S = BlockSparseSolver(
    H, beta, w_max, eps, gf_struct, 
    conserved_operators=[N_tot],
    )

S.Delta_tau['up'] << Delta_tau
S.Delta_tau['do'] << Delta_tau

S.solve(max_order=order, tol=ppsc_tol, maxiter=ppsc_maxiter)

#if is_root():
if False:

    from triqs.gf import make_gf_imtime
    G_tau_fine = make_gf_imtime(S.G_tau, n_tau=400)

    def abs_block_gf(G):
        for bidx, g in G:
            g.data[:] = np.abs(g.data)
        return G

    S.G_tau = abs_block_gf(S.G_tau)
    G_tau_fine = abs_block_gf(G_tau_fine)

    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
    oplotr(S.G_tau)
    oplotr(G_tau_fine)
    plt.semilogy([], [])
    plt.show()