

import numpy as np

from triqs.gfs import Gf, MeshDLRImTime, MeshImTime, make_gf_imtime
from triqs.operators import c, c_dag, Operator

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs.utility import mpi
from triqs_xca.block_sparse_solver import BlockSparseSolver

t = -np.sqrt(2.0)
H = t * ( c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0) )

fundamental_operators = [c('0', 0), c('0', 1)]

beta = 4.0

ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

#m = MeshDLRImTime(beta=beta, statistic='Fermion', eps=1e-12, w_max=2.0)
m = MeshImTime(beta=beta, statistic='Fermion', n_tau=400)

G_tau = Gf(mesh=m, target_shape=[])
ed.set_g2_tau(G_tau, c('0', 0), c_dag('0', 0))

# -- XCA approximation

H0 = 0. * c_dag('0', 0) * c('0', 0)

Ss = []

orders = [1]
for order in orders:
    S = BlockSparseSolver(H_loc=H0, beta=beta, w_max=4.0, eps=1e-10, gf_struct=[['0', 1]])
    S.Delta_tau['0'].data[:] = -0.5 * t**2
    
    S.solve_bare(max_order=order, use_dyson=False, verbose=True)
    S.G_tau_bare = S.G_tau.copy()
    S.eta = 0

    S.solve_bare(max_order=order, use_dyson=True, verbose=True)
    S.G_tau_bare_dyson = S.G_tau.copy()

    S.G = S.G0.copy() # reset
    S.eta = 0.0 # reset
    S.solve(max_order=order, tol=1e-8, maxiter=20, verbose=True, mix=1.)
    Ss.append(S)

from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

if mpi.is_master_node():
    for S in Ss:
        if S.max_order % 2 == 1:
            s = '-'
        else:
            s = ':'

        n_tau = 40
        oplot(make_gf_imtime(S.G_tau_bare, n_tau=n_tau).real, '.'+s, label=f'O{S.max_order} bare')
        oplot(make_gf_imtime(S.G_tau_bare_dyson, n_tau=n_tau).real, s+'x', label=f'O{S.max_order} bare Dys')
        oplot(make_gf_imtime(S.G_tau, n_tau=n_tau).real, s+'+', label=f'O{S.max_order} sc')

    oplot(G_tau.real, '--', lw=4., alpha=0.5, label='ED')
    #plt.ylim(top=0.)
    plt.show()