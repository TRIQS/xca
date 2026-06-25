

import numpy as np

from triqs.gfs import BlockGf, Gf, MeshDLRImTime, MeshImTime, make_gf_imtime, make_gf_dlr
from triqs.operators import c, c_dag, Operator

from triqs.utility import mpi

from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse import convolve_ppsc as conv


def diag_blockgf(tau_mesh, vec_tau):

    G = BlockGf(name_list=['0'], block_list=[Gf(mesh=tau_mesh, target_shape=[2, 2])])
    for idx in range(G['0'].target_shape[0]):
        G['0'].data[:, idx, idx] = vec_tau
    return G


class DimerAnalyticSolution:

    def __init__(self, tau_mesh, V):
        self.V = V
        self.tau_mesh = tau_mesh
        self.beta = tau_mesh.beta

        beta = self.beta
        
        self.eta0 = np.log(2) / beta
        self.eta = np.log(1 + np.cosh(beta * V)) / beta
        self.deta = np.log(np.cosh(beta * V / 2)) * 2 / beta

        np.testing.assert_almost_equal(self.deta, self.eta - self.eta0)

        tau = np.array([ float(t) for t in tau_mesh ])

        self.G0 = diag_blockgf(tau_mesh, -np.exp(-self.eta0 * tau) )

        self.rho0 = -make_gf_dlr(self.G0['0'])(beta)
        self.norm0 = np.trace(self.rho0).real
        np.testing.assert_almost_equal(self.norm0, 1.0)

        self.G  = diag_blockgf(tau_mesh, - 0.5 * np.exp(-self.eta * tau) * (1 + np.cosh(V * tau)) )

        self.rho = -make_gf_dlr(self.G['0'])(beta)
        self.norm = np.trace(self.rho).real
        np.testing.assert_almost_equal(self.norm, 1.0)

        self.Sigma = diag_blockgf(tau_mesh, -V**2/2 * np.exp(-self.eta * tau) * np.cosh(V*tau / np.sqrt(2)) )
        self.Sigma0 = diag_blockgf(tau_mesh, - np.exp(-self.eta * tau) * (
            (V - self.deta)**2/4 * np.exp(+tau * V) + (V + self.deta)**2/4 * np.exp(-tau * V) + self.deta**2/2 ) )

        # Test Dyson G = G0 + G0 * ( deta + Sigma ) * G
        G_ref_bold = self.G0 + conv(self.G0, self.G * self.deta) + conv(self.G0, conv(self.Sigma, self.G))
        np.testing.assert_array_almost_equal(G_ref_bold['0'].data, self.G['0'].data)

        # Test (bare) Dyson G = G0 + G0 * ( deta + Sigma_0 ) * G_0
        G_ref_bare = self.G0 + conv(self.G0, self.G0 * self.deta) + conv(self.G0, conv(self.Sigma0, self.G0))
        np.testing.assert_array_almost_equal(G_ref_bare['0'].data, self.G['0'].data)
        
        self.spgf = Gf(mesh=tau_mesh, target_shape=[1, 1])
        self.spgf.data[:, 0, 0] = -0.5*(np.exp(-V*tau)/(1+np.exp(-beta*V)) + np.exp(+V*tau)/(1+np.exp(beta*V)))        


class DimerED:

    def __init__(self, tau_mesh, V):

        self.tau_mesh = tau_mesh
        self.V = V

        m = tau_mesh
        
        H = V * ( c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0) )

        Sd = BlockSparseSolver(
            H_loc=H, beta=m.beta, w_max=m.w_max, eps=m.eps, 
            gf_struct=[['0', 2]], conserved_operators=[])

        Sd.solve(max_order=0, spgf_max_order=1, maxiter=0, verbose=True)
         
        G_00 = Sd.G0['0'][0, 0]
        G_01 = Sd.G0['0'][1, 1]
        
        self.G = diag_blockgf(tau_mesh, 0.5 * (G_00 + G_01).data )

        eta0 = np.log(2) / beta
        tau = np.array([ float(t) for t in tau_mesh ])
        self.G['0'].data[:] *= np.exp(+eta0 * tau)[:, None, None]

        self.rho = -make_gf_dlr(self.G['0'])(beta)
        self.norm = np.trace(self.rho).real
        
        self.spgf = Gf(mesh=tau_mesh, target_shape=[1, 1])
        self.spgf.data[:, 0, 0] = Sd.G_tau['0'][0, 0].data


class DimerXCA:

    def __init__(self, tau_mesh, V, order=1, bold=True):

        self.tau_mesh = tau_mesh
        self.V = V
        
        self.order = order

        m = tau_mesh
        
        S = BlockSparseSolver(
            H_loc=0.*c_dag('0', 0)*c('0', 0),
            beta=m.beta, w_max=m.w_max, eps=m.eps, 
            gf_struct=[['0', 1]],
            conserved_operators=[],
            )

        S.Delta_tau['0'].data[:] = -0.5 * V**2

        if bold:
            S.solve(max_order=order, verbose=True, tol=1e-12, hyb_tol=1e-9)
        else:
            S.solve_bare(max_order=order, verbose=True, hyb_tol=1e-9)

        self.eta = S.eta
        self.G = S.G.copy()
        self.Sigma = S.Sigma.copy()
        self.sgpf = S.G_tau.copy()
        

        
V = 2.5     
beta = 4/3
w_max = 5.0
eps = 1e-10

tau_mesh = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)

anal = DimerAnalyticSolution(tau_mesh, V)
ed = DimerED(tau_mesh, V)

#np.testing.assert_array_almost_equal(anal.G['0'].data, ed.G['0'].data)
np.testing.assert_array_almost_equal(anal.spgf.data, ed.spgf.data)

bold = DimerXCA(tau_mesh, V, order=3, bold=True)
bare = DimerXCA(tau_mesh, V, order=3, bold=False)

print(f'norm = {anal.norm} (anal)')
print(f'norm = {ed.norm} (ed)')


print(f'anal.eta = {anal.eta}')
print(f'bold.eta = {bold.eta}')


from triqs.plot.mpl_interface import oplot, plt

plt.figure(figsize=(3.25*2, 6))

subp = [2, 1, 1]

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime(anal.G0['0'][0, 0], n_tau=100).real, label='G0 anal')
oplot(make_gf_imtime(anal.G['0'][0, 0], n_tau=100).real, label='G anal')
oplot(make_gf_imtime(ed.G['0'][0, 0], n_tau=100).real, ':', label='ed')

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime(ed.G['0'][0, 0], n_tau=100).real, '-', label='ed')
oplot(make_gf_imtime(bare.G['0'][0, 0], n_tau=100).real, '--', label='O1 (bare)')
oplot(make_gf_imtime(bold.G['0'][0, 0], n_tau=100).real, ':', label='O1 (bold)')

plt.ylim(top=0)

plt.tight_layout()
plt.show()

exit()

# -- Full dimer calculation

t = -np.sqrt(2.0)

H = t * ( c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0) )

fundamental_operators = [c('0', 0), c('0', 1)]

Sd = BlockSparseSolver(
    H_loc=H, beta=beta, w_max=w_max, eps=eps, 
    gf_struct=[['0', 2]], conserved_operators=[])

Sd.solve(max_order=0, spgf_max_order=1, maxiter=0, verbose=True)

rho_d = Sd.many_body_density_matrix(Sd.G0)[0][1]
print(f'rho_d =\n{rho_d.real}')

tau = np.array([ float(t) for t in Sd.G.mesh ])

G_tau_ref = Sd.G_tau['0'][0, 0]

G_tau_anal_ref = G_tau_ref.copy()
G_tau_anal_ref.data[:] = -0.5*(np.exp(-t*tau)/(1+np.exp(-beta*t)) + np.exp(+t*tau)/(1+np.exp(beta*t)))
np.testing.assert_array_almost_equal(G_tau_ref.data, G_tau_anal_ref.data)

g_tau_anal_ref = -0.5 * np.cosh(t*(tau - beta/2)) / np.cosh(t*beta/2)
np.testing.assert_array_almost_equal(g_tau_anal_ref, G_tau_anal_ref.data)

#exit()

H_ref = np.zeros((4, 4))
H_ref[1, 2] = t
H_ref[2, 1] = t
print(f'H_ref =\n{H_ref}')

E, U = np.linalg.eigh(H_ref)
E_ref = np.array([t, 0, 0, -t])
print(f'E     = {E}')
print(f'E_ref = {E_ref}')
np.testing.assert_array_almost_equal(E, E_ref)

print(f'U = \n{U}')

Z = np.sum(np.exp(-beta * E))
Z_ref = 2 * ( 1 +  np.cosh(beta * t) )
print(f'Z     = {Z}')
print(f'Z_ref = {Z_ref}')
np.testing.assert_array_almost_equal(Z, Z_ref)

eta = - np.log(Z) / beta
eta_ref = - np.log(2) / beta - np.log(1 + np.cosh(beta * t)) / beta

print(f'eta     = {eta}')
print(f'eta_ref = {eta_ref}')

G_0_ref = BlockGf(name_list=['0'], block_list=[Gf(mesh=Sd.G.mesh, target_shape=[4, 4])])

G_0_ref['0'].data[:] = - np.einsum('ab,tb,bc->tac', U, np.exp(-(E[None, :] - eta) * tau[:, None]), U.T.conj())

g_anal_00 = -np.exp(-(0 - eta)*tau)
g_anal_11 = -0.5*( np.exp(-(t - eta)*tau) + np.exp(-(-t - eta)*tau) )
g_anal = 0.5 * (g_anal_00 + g_anal_11)
g_anal_ref = - 0.5 * np.exp(eta * tau) * ( 1 + np.cosh(t * tau) )

np.testing.assert_array_almost_equal(g_anal, g_anal_ref)

np.testing.assert_array_almost_equal(g_anal_00, Sd.G['0'][0, 0].data)
np.testing.assert_array_almost_equal(g_anal_11, Sd.G['0'][1, 1].data)

G_ref_anal = BlockGf(name_list=['0'], block_list=[Gf(mesh=Sd.G.mesh, target_shape=[2, 2])])
G_ref_anal['0'][0, 0].data[:] = 0.5 * (g_anal_00 + g_anal_11)
G_ref_anal['0'][1, 1].data[:] = 0.5 * (g_anal_00 + g_anal_11)



if False:
    from triqs.plot.mpl_interface import oplot, plt
    oplot(make_gf_imtime(Sd.G, n_tau=100).real)
    oplot(G_0_ref, '--', alpha=0.5)
    plt.plot(tau, g_anal_00, 'o')
    plt.plot(tau, g_anal_11, 's')
    plt.show()

    exit()


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

def detect_zero_topology(topology):
    n = 2*len(topology)
    diffs = np.zeros(n)
    diffs[-1] = -1

    for arc in topology:
        s, f = arc
        d = diffs[s - 1]
        diffs[s] = -d
        diffs[f] = +d

    occs = np.cumsum(diffs)
    #print(f'diffs = {diffs}')
    #print(f'occs = {occs}')
    return (occs <= 1).all() and (occs >= 0).all()


def eval_pseudo_particle_self_energy_order(S, G, order, connected=True):

    import time

    Sigma = S.get_zero_pseudo_particle_propagator()
    
    from triqs_xca.diag import all_pairings, all_connected_pairings
    pairings = all_connected_pairings if connected else all_pairings

    C_N = 0
    
    for sign, topology in pairings(order):
        non_zero = detect_zero_topology(topology)
        #print(f'non_zero = {non_zero}')
        if not non_zero:
            continue
        
        topology_int32 = np.array(topology, dtype=np.int32)
        t1 = time.time()
        Sigma_diag = pow(-1, order) * sign * \
            S._BlockSparseSolver__eval_pseudo_particle_self_energy_topology_loop(G, topology_int32, verbose=True)
        abs_max = np.max(np.abs(Sigma_diag['0'].data))
        print(f'SIGMA: O{order} topo {topology} sign {sign:+d} absmax {abs_max:2.2E} time {time.time() - t1} s')

        if abs_max > 0:
            assert( non_zero )
            sign = np.sign(Sigma_diag['0'].data[-1, 0, 0].real)
            C_N -= sign
            Sigma += Sigma_diag
        else:
            assert( not non_zero )

    return Sigma, C_N

# -- Vary order

# -- Analyze self-energy diagrams

orders = [1, 2, 3, 4]
Ss = []

#Sigma_old = None

for order in orders:
    H0 = 0. * c_dag('0', 0) * c('0', 0)
    S = BlockSparseSolver(
        H_loc=H0, beta=beta, w_max=w_max, eps=eps, 
        gf_struct=[['0', 1]],
        conserved_operators=[],
        )
    
    S.Delta_tau['0'].data[:] = -0.5 * t**2

    S.solve(max_order=order, verbose=True, tol=1e-12, hyb_tol=1e-9)
    rho = S.many_body_density_matrix(S.G)[0][1]
    print(f'rho =\n{rho.real}')
    Ss.append(S)

    
orders_sigma = [1, 2, 3, 4]
Ss_sigma = []
    
for order in orders_sigma:
    H0 = 0. * c_dag('0', 0) * c('0', 0)
    S = BlockSparseSolver(
        H_loc=H0, beta=beta, w_max=w_max, eps=eps, 
        gf_struct=[['0', 1]],
        conserved_operators=[],
        )
    S.Delta_tau['0'].data[:] = -0.5 * t**2
    S.solve(max_order=2, spgf_max_order=1, maxiter=0, verbose=True)
    S.max_order = order
    S.Sigma0, S.C_N = eval_pseudo_particle_self_energy_order(S, S.G0, order)
    print(f'N = {order}, C_N = {S.C_N}')
    Ss_sigma.append(S)
        

from triqs.plot.mpl_interface import oplot, plt

plt.figure(figsize=(3.25*2, 9))

subp = [5, 2, 1]

plt.subplot(*subp); subp[-1] += 1
oplot(make_gf_imtime(G_tau_ref, n_tau=100).real, label='exact')
for S in Ss:
    oplot(make_gf_imtime(S.G_tau, n_tau=100).real, ':', label=f'O{S.max_order}')
plt.ylim(top=0)

plt.subplot(*subp); subp[-1] += 1
for S in Ss:
    diff = (S.G_tau['0'][0, 0] - G_tau_ref)
    diff.data[:] = np.abs(diff.data)
    oplot(make_gf_imtime(diff, n_tau=100).real, label=f'O{S.max_order}')
plt.semilogy([], [])

#plt.subplot(*subp); subp[-1] += 1
#oplot(make_gf_imtime(Sd.G0, n_tau=100).real, label=None)

plt.subplot(*subp); subp[-1] += 1
#oplot(make_gf_imtime(S.G0, n_tau=100).real, label='G0')
oplot(make_gf_imtime(G_ref['0'][0, 0], n_tau=20).real, '.', label='G (ref)')
for S in Ss:
    oplot(make_gf_imtime(S.G['0'][0, 0], n_tau=100).real, label=f'O{S.max_order}')

plt.subplot(*subp); subp[-1] += 1
for S in Ss: 
    diff = (S.G - G_ref)['0']
    diff.data[:] = np.abs(diff.data)
    oplot(make_gf_imtime(diff[0, 0], n_tau=100).real, label=f'O{S.max_order}')
plt.ylabel('Error ppgf')
plt.semilogy([], [])

C_N = [-1, 0, 1, 2, 1]

plt.subplot(*subp); subp[-1] += 1

plt.plot(np.ones(len(C_N)) * beta, C_N, 'or')
for S in Ss_sigma:
    #plt.subplot(*subp); subp[-1] += 1
    SoG = S.Sigma0['0'][0, 0].copy()
    N = S.max_order
    fac = np.prod(np.arange(1, 2*N-2 + 1))
    SoG.data[:] /= S.G0['0'][0, 0].data * (-t**2/2)**N / fac
    c = plt.plot([], [])[0].get_color()
    oplot(make_gf_imtime(SoG, n_tau=100).real, label=f'O{S.max_order}', color=c)
    plt.plot(beta, S.C_N, 'xb')
    plt.ylabel('Sigma[G0]')

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
