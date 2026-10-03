r""" Analytic test of diagrammatics for two interacting fermionic orbitals

Single-site model with Hamiltonian

..math::
    H = -\mu_0 n_0 - \mu_1 n_1 + U n_0 n_1

coupled to a bath of one pole :math:`e_a` per orbital,

..math::
    \Delta_{ab}(\tau) = \delta_{ab} K(\tau, e_a), \quad
    K(\tau, \omega) = -\frac{e^{-\omega \tau}}{1 + e^{-\beta \omega}}.

The different :math:`\mu_a` and :math:`e_a` make the orbitals distinguishable. The local
Hamiltonian is diagonal in the Fock basis (no hopping), so the test covers the diagram values
and signs but not the atom_diag eigenvector rotation.

The atomic states have energies :math:`E_s = (0, -\mu_0, -\mu_1, U - \mu_0 - \mu_1)`, and the
pseudo particle Green's function is a single exponential per state

..math::
    G_s(\tau) = -e^{-\tau (E_s + \eta_0)}, \quad \eta_0 = \beta^{-1} \ln Z_{\mathrm{at}},

such that :math:`Z = 1`. With these propagators, and the exponential hybridization
function, every self-energy and single-particle Green's function diagram is a
sum over atomic state paths of nested integrals of exponentials. The integrals are
evaluated exactly (matrix exponential of a bidiagonal matrix) and compared,
topology by topology up to third order, with the block-sparse solver
for all choices of conserved operators.

Conventions of the analytic diagrams:

- A self-energy topology with :math:`n` pairs has vertices :math:`0, \ldots, 2n-1`
  at times :math:`0 = t_0 < \ldots < t_{2n-1} = \tau`.
- A single-particle Green's function topology with :math:`n` pairs has vertices
  :math:`0, \ldots, 2n-1` on the closed contour :math:`[0, \beta]`. The pair containing
  vertex 0 is the external pair, :math:`c^\dagger` at time 0 and :math:`c` at time :math:`\tau`,
  the other :math:`n-1` pairs are hybridization lines.
- A hybridization line between vertices :math:`i < j` contributes
  :math:`K(t_j - t_i, e_a)` if vertex :math:`i` annihilates and :math:`K(t_j - t_i, -e_a)` if it creates,
  with :math:`a` the orbital of the pair.
- Each pseudo particle propagator contributes a factor :math:`G_s`, and the
  operators their fermionic matrix elements.
"""

import itertools

import numpy as np

from scipy.linalg import expm

from triqs.gfs import Gf, MeshDLRImFreq, inverse, iOmega_n, make_gf_dlr_imtime
from triqs.operators import n

from triqs_xca.diag import all_connected_pairings
from triqs_xca import Solver
from triqs_xca.solver import hamiltonian_matrix, pseudo_particle_block_gf_to_dense


# -- Analytic reference

def fock_operators():
    """ Matrices of c_a and c^dagger_a in the basis |n_0 n_1> = |00>, |10>, |01>, |11> """

    occ = [(0, 0), (1, 0), (0, 1), (1, 1)]
    ops = dict()

    for a, create in itertools.product(range(2), (True, False)):
        M = np.zeros((4, 4))
        for s, o in enumerate(occ):
            if o[a] == (0 if create else 1):
                o_new = list(o)
                o_new[a] = 1 - o_new[a]
                M[occ.index(tuple(o_new)), s] = (-1)**sum(o[:a])
        ops[(a, create)] = M

    return ops


def exp_simplex_integral(lam, T):
    """ Integral of exp(sum_k lam[k] * s[k]) over 0 < s[0] < ... < s[m-1] < T for each T in an array

    The nested integral J_k(t) = int_0^t exp(lam[k-1] s) J_{k-1}(s) ds is
    J_k(t) = exp(L_k t) v_k(t) with L_k = lam[0] + ... + lam[k-1] and
    v' = A v, v(0) = e_0, where A is lower bidiagonal. """

    m = len(lam)
    if m == 0:
        return np.ones_like(T)

    L = np.cumsum(lam)
    A = np.zeros((m + 1, m + 1))
    for k in range(1, m + 1):
        A[k, k] = -L[k-1]
        A[k, k-1] = 1.

    return np.array([ expm(A * t)[m, 0] * np.exp(L[-1] * t) for t in T ])


class AnalyticTwoFermions:

    def __init__(self, beta, mu, U, e1):
        """ mu and e1 are per orbital """

        self.beta = beta
        self.e1 = e1
        self.ops = fock_operators()

        self.E = np.array([0., -mu[0], -mu[1], U - mu[0] - mu[1]])
        self.eta0 = np.log(np.sum(np.exp(-beta * self.E))) / beta
        self.eps = self.E + self.eta0 # energies of the normalized propagators


    def G(self, tau):
        """ Pseudo particle Green's function, diagonal in the Fock basis, shape (tau, 4) """
        return -np.exp(-np.outer(tau, self.eps))


    def line_weight(self, orbital, creates_first):
        """ Pole of the hybridization line and its prefactor, K(t, w) = pref * exp(-w t) """

        w = -self.e1[orbital] if creates_first else self.e1[orbital]
        return w, -1. / (1. + np.exp(-self.beta * w))


    def operator_paths(self, topology, ext_first_creates=None):
        """ Yield (orbital, creates) per vertex, initial state, state path and matrix element of
        all non-vanishing operator assignments. If given, the first vertex pair is the external
        pair and its first vertex creates (True) or annihilates (False). """

        nv = 2 * len(topology)
        pairs = list(topology)

        choices = []
        for p in range(len(pairs)):
            first_types = (True, False) if (p > 0 or ext_first_creates is None) else (ext_first_creates,)
            choices.append(list(itertools.product(range(2), first_types)))

        for assignment in itertools.product(*choices):
            orbital, creates = [None] * nv, [None] * nv
            for (i, j), (a, c) in zip(pairs, assignment):
                orbital[i] = orbital[j] = a
                creates[i], creates[j] = c, not c

            Ms = [ self.ops[(orbital[v], creates[v])] for v in range(nv) ]

            for s0 in range(4):
                vec = np.zeros(4)
                vec[s0] = 1.
                path = [s0]
                for M in Ms:
                    vec = M @ vec
                    if not vec.any(): break
                    path.append(int(np.argmax(np.abs(vec))))
                else:
                    yield orbital, creates, s0, path, vec[path[-1]]


    def self_energy_topology(self, topology, tau):
        """ Self-energy diagram, shape (tau, 4, 4) """

        nv = 2 * len(topology)
        Sigma = np.zeros((len(tau), 4, 4))

        for orbital, creates, s0, path, mel in self.operator_paths(topology):

            lam = np.zeros(nv)
            pref = mel * (-1.)**(nv - 1)

            for k in range(nv - 1): # propagators between vertex k and k + 1
                e = self.eps[path[k+1]]
                lam[k] += e
                lam[k+1] -= e

            for i, j in topology:
                w, p = self.line_weight(orbital[i], creates[i])
                lam[i] += w
                lam[j] -= w
                pref *= p

            Sigma[:, path[-1], s0] += \
                pref * np.exp(lam[-1] * tau) * exp_simplex_integral(lam[1:-1], tau)

        return Sigma


    def spgf_topology(self, topology, tau):
        """ Single-particle Green's function diagram, shape (tau, 2, 2) """

        nv = 2 * len(topology)
        k = next( j for i, j in topology if i == 0 ) # vertex at time tau
        spgf = np.zeros((len(tau), 2, 2))

        for orbital, creates, s0, path, mel in self.operator_paths(topology, ext_first_creates=True):

            if path[-1] != s0: continue

            lam = np.zeros(nv + 1) # the last entry is the coefficient of the time beta
            pref = mel * (-1.)**nv

            for p in range(nv): # propagators between vertex p and p + 1 (cyclic)
                e = self.eps[path[p+1]]
                lam[p] += e
                lam[p+1] -= e

            for i, j in topology[1:]:
                w, p = self.line_weight(orbital[i], creates[i])
                lam[i] += w
                lam[j] -= w
                pref *= p

            left, right = lam[1:k], lam[k+1:nv]

            spgf[:, orbital[0], orbital[0]] += \
                pref * np.exp(lam[nv] * self.beta) * np.exp((lam[k] + right.sum()) * tau) * \
                exp_simplex_integral(left, tau) * exp_simplex_integral(right, self.beta - tau)

        return spgf


# -- Test

def can_plot():
    """ Plot only with an interactive matplotlib backend (i.e. with a display) """
    import matplotlib
    return matplotlib.get_backend().lower() in [ b.lower() for b in matplotlib.rcsetup.interactive_bk ]


def plot_comparison(tau, G_BSS, G_ana, results):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for s in range(4):
        plt.plot(tau, G_BSS[:, s, s], 'x', color=f'C{s}')
        plt.plot(tau, G_ana[:, s], '-', color=f'C{s}', label=f'state {s}')
    plt.plot([], [], 'x', color='gray', label='Block-sparse')
    plt.plot([], [], '-', color='gray', label='Analytic')
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_0(\tau)$')
    plt.tight_layout()

    plt.figure(figsize=(14, 12))
    subp = [6, 4, 1]
    for topology, r in results.items():
        for name, bss, ana in [('\\Sigma', r['Sigma_BSS'], r['Sigma_ana']), ('g', r['spgf_BSS'], r['spgf_ana'])]:

            plt.subplot(*subp); subp[-1] += 1
            for i, j in itertools.product(range(ana.shape[-1]), repeat=2):
                if np.abs(ana[:, i, j]).max() > 0 or np.abs(bss[:, i, j]).max() > 0:
                    plt.plot(tau, bss[:, i, j], 'x')
                    plt.plot(tau, ana[:, i, j], '-')
            plt.xlabel(r'$\tau$')
            plt.ylabel(f'${name}_{{{topology}}}(\\tau)$', fontsize=7)

            plt.subplot(*subp); subp[-1] += 1
            for i, j in itertools.product(range(ana.shape[-1]), repeat=2):
                diff = bss[:, i, j] - ana[:, i, j]
                if np.abs(diff).max() > 0:
                    plt.plot(tau, diff, 'o-')
            plt.xlabel(r'$\tau$')
            plt.ylabel('Difference')

    plt.tight_layout()
    plt.savefig('figure_xca_top_cf.pdf')
    plt.show()


def test_diagrams_block_sparse_vs_analytic(e1=(+1.5, -0.7), beta=2.0, conserved_operators='none', verbose=False):

    print('='*72)
    print(f'beta = {beta}, e1 = {e1}, conserved_operators = {conserved_operators}')
    print('='*72)

    # -- Parameters

    mu = (0.3, 0.55)
    U = 3.0
    e1 = np.array(e1)

    eps = 1e-12
    w_max = 20.0

    # Tolerance of the hybridization fit, the analytic diagrams are exact
    atol = 100 * eps

    # -- Local Hamiltonian

    gf_struct = [['0', 2]]

    N_0 = n('0', 0)
    N_1 = n('0', 1)
    N_op = N_0 + N_1

    H = -mu[0] * N_0 - mu[1] * N_1 + U * N_0 * N_1

    conserved_operators = dict(
        none=[],
        total_density=[N_op],
        individual_density=[N_0, N_1],
        automatic='automatic',
        )[conserved_operators]

    # -- Hybridization function

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps, symmetrize=False)
    Delta_w = Gf(mesh=mesh_w, target_shape=[2]*2)
    for a in range(2):
        Delta_w[a, a] << inverse(iOmega_n - e1[a])

    Delta_tau = make_gf_dlr_imtime(Delta_w)

    # -- Block sparse solver

    BSS = Solver(
        H, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators,
        )

    BSS.Delta_tau['0'] << Delta_tau

    BSS.fit_hybridization(tol=100*eps, compression=True, verbose=verbose)
    BSS.init_diagram_evaluator()

    tau = np.array([ float(t) for t in BSS.mesh_tau ])
    ana = AnalyticTwoFermions(beta, mu, U, e1)

    # -- Atomic Hamiltonian and pseudo particle Green's function

    H_mat = hamiltonian_matrix(BSS.atom_diag)
    np.testing.assert_allclose(H_mat, np.diag(ana.E), rtol=0, atol=1e-12)

    G_BSS = pseudo_particle_block_gf_to_dense(BSS.pseudo_particle_greens_function(), BSS.atom_diag)
    G_ana = ana.G(tau)

    G_ref = np.einsum('ts,sr->tsr', G_ana, np.eye(4))
    print(f'G_diff = {np.max(np.abs(G_BSS.data - G_ref)):2.2E}')
    np.testing.assert_allclose(G_BSS.data, G_ref, rtol=0, atol=1e-12)

    Z = BSS.partition_function()
    print(f'Z = {Z}')
    np.testing.assert_allclose(Z, 1, rtol=0, atol=atol)

    # Solving Dyson with zero self-energy gives back the atomic propagator
    G_DYSON_BSS = BSS.solve_dyson(BSS.Sigma, BSS.eta)
    G_DYSON_BSS = pseudo_particle_block_gf_to_dense(G_DYSON_BSS, BSS.atom_diag)
    print(f'G_dyson_diff = {np.max(np.abs(G_DYSON_BSS.data - G_BSS.data)):2.2E}')
    np.testing.assert_allclose(G_DYSON_BSS.data, G_BSS.data, rtol=0, atol=atol)

    # -- Self-energy and single-particle Green's function topologies

    results = dict()

    # Sums over all topologies up to the current order, with the diagram weights (-1)^order * sign
    Sigma_sum_ana = np.zeros((len(tau), 4, 4))
    spgf_sum_ana = np.zeros((len(tau), 2, 2))

    for order in [1, 2, 3]:
        print(f'order = {order}')

        for sign, topology in all_connected_pairings(order):

            print(f'  topology = {topology}')

            # The topology evaluators leave out the topology sign, and the self-energy evaluator also a factor (-1)^(order + 1)
            Sigma_BSS = BSS.eval_pseudo_particle_self_energy_topology(BSS.G, np.array(topology, dtype=np.int32))
            Sigma_BSS = (-1)**(order + 1) * sign * pseudo_particle_block_gf_to_dense(Sigma_BSS, BSS.atom_diag).data
            Sigma_ana = ana.self_energy_topology(topology, tau)

            spgf_BSS = sign * BSS.eval_single_particle_greens_function_topology(BSS.G, np.array(topology, dtype=np.int32)).data
            spgf_ana = ana.spgf_topology(topology, tau)

            print(f'    Sigma_diff = {np.max(np.abs(Sigma_BSS.real - Sigma_ana)):2.2E}')
            print(f'    spgf_diff = {np.max(np.abs(spgf_BSS.real - spgf_ana)):2.2E}')

            np.testing.assert_allclose(Sigma_BSS.real, Sigma_ana, rtol=0, atol=atol)
            np.testing.assert_allclose(spgf_BSS.real, spgf_ana, rtol=0, atol=atol)

            # The imaginary parts vanish for real Hamiltonian and hybridization
            np.testing.assert_allclose(Sigma_BSS.imag, 0, rtol=0, atol=atol)
            np.testing.assert_allclose(spgf_BSS.imag, 0, rtol=0, atol=atol)

            # Nothing outside of the diagonal in the Fock basis for an orbital diagonal hybridization
            np.testing.assert_allclose(Sigma_BSS.real, np.einsum('tss->ts', Sigma_BSS.real)[:, :, None] * np.eye(4), atol=atol)
            np.testing.assert_allclose(spgf_BSS.real[:, 0, 1], 0, atol=atol)

            Sigma_sum_ana += (-1)**order * sign * Sigma_ana
            spgf_sum_ana += (-1)**order * sign * spgf_ana
            results[tuple(topology)] = dict(
                Sigma_BSS=Sigma_BSS.real, Sigma_ana=Sigma_ana, spgf_BSS=spgf_BSS.real, spgf_ana=spgf_ana)

        # -- Sums over all topologies up to the current order

        Sigma_BSS = pseudo_particle_block_gf_to_dense(BSS.eval_pseudo_particle_self_energy(BSS.G, order), BSS.atom_diag)
        spgf_BSS = BSS.eval_single_particle_greens_function(BSS.G, order)

        np.testing.assert_allclose(Sigma_BSS.data.real, Sigma_sum_ana, rtol=0, atol=atol)
        np.testing.assert_allclose(spgf_BSS.data.real, spgf_sum_ana, rtol=0, atol=atol)
        np.testing.assert_allclose(Sigma_BSS.data.imag, 0, rtol=0, atol=atol)
        np.testing.assert_allclose(spgf_BSS.data.imag, 0, rtol=0, atol=atol)
        print('  Passed')

    # -- Vizualize

    if verbose and can_plot():
        plot_comparison(tau, G_BSS.data.real, G_ana, results)


if __name__ == '__main__':

    ops = [
        'none',
        'total_density',
        'individual_density',
        'automatic',
        ]

    for e1 in [(+1.5, -0.7), (-1.5, +0.7)]:
        for op in ops:
            test_diagrams_block_sparse_vs_analytic(
                e1=e1, beta=2.0, conserved_operators=op, verbose=False)
