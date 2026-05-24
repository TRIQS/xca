

import numpy as np

from triqs.gfs import BlockGf, Gf, MeshDLRImTime, MeshImTime, make_gf_imtime, make_gf_dlr
from triqs.operators import c, c_dag, Operator

from triqs.utility import mpi
from triqs_xca.block_sparse_solver import BlockSparseSolver


beta = 1.0

# -- Full dimer calculation

t = -np.sqrt(2.0)

H = t * ( c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0) )

fundamental_operators = [c('0', 0), c('0', 1)]

Sd = BlockSparseSolver(
    H_loc=H, beta=beta, w_max=4.0, eps=1e-10, 
    gf_struct=[['0', 2]], conserved_operators=[])
Sd.solve(max_order=0, spgf_max_order=1, maxiter=0, verbose=True)

rho_d = Sd.many_body_density_matrix(Sd.G0)[0][1]
print(f'rho_d =\n{rho_d.real}')

G_tau_ref = Sd.G_tau['0'][0, 0]


# -- Trace out one site

# The reduced ppgf is given by the diagonal trace over the bath
# up to an additional normalization Tr[G(beta)] = -1

G_00 = Sd.G0['0'][0, 0]
G_01 = Sd.G0['0'][1, 1]

#G_ref = S.G.copy()
G_ref = BlockGf(name_list=['0'], block_list=[Gf(mesh=Sd.G.mesh, target_shape=[2, 2])])
G_ref['0'][0, 0] = 0.5 * (G_00 + G_01)
G_ref['0'][1, 1] = 0.5 * (G_00 + G_01)

# Normalize G_test
rho_ref = -make_gf_dlr(G_ref['0'])(beta)
print(f'rho_ref =\n{rho_ref}')

eta = np.log(np.trace(rho_ref)) / beta
tau = np.array([float(t) for t in G_ref.mesh])

G_ref_norm = G_ref.copy()
G_ref_norm['0'].data[:] *= np.exp(- eta * tau)[:, None, None]
rho_ref_norm = -make_gf_dlr(G_ref_norm['0'])(beta)

print(f'rho_ref_norm =\n{rho_ref_norm}')
#exit()

G_ref = G_ref_norm


# -- Perturbative calculation -- Expanding in one site

H0 = 0. * c_dag('0', 0) * c('0', 0)
S = BlockSparseSolver(H_loc=H0, beta=beta, w_max=4.0, eps=1e-10, 
                      gf_struct=[['0', 1]], conserved_operators=[])
S.Delta_tau['0'].data[:] = -0.5 * t**2
S.solve(max_order=1, verbose=True)
rho = S.many_body_density_matrix(S.G)[0][1]
print(f'rho =\n{rho.real}')


from triqs.plot.mpl_interface import oplot, plt

plt.figure(figsize=(3.25*2, 8))

subp = [2, 2, 1]

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime(G_tau_ref, n_tau=100).real, label='exact')
oplot(make_gf_imtime(S.G_tau, n_tau=100).real, ':', label='o1')
plt.ylim(top=0)

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime(G_tau_ref - S.G_tau['0'][0, 0], n_tau=100).real, label='exact')

#plt.subplot(*subp); subp[-1] += 1
#oplot(make_gf_imtime(Sd.G0, n_tau=100).real, label=None)

plt.subplot(*subp); subp[-1] += 1
#oplot(make_gf_imtime(S.G0, n_tau=100).real, label='G0')
oplot(make_gf_imtime(S.G['0'][0, 0], n_tau=100).real, label='G')
oplot(make_gf_imtime(G_ref['0'][0, 0], n_tau=20).real, '.', label='G (ref)')

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime((S.G - G_ref)['0'][0, 0], n_tau=100).real, label=None)
plt.ylabel('Error ppgf')

if False:
    subp = [3, 1, 1]
    for bidx in ['0', '1', '2']:
        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_imtime(Sd.G0[bidx], n_tau=100).real)
        plt.title(bidx)
plt.tight_layout()
plt.show()

exit()



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