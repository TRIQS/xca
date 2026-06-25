

import numpy as np

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_dlr, inverse, iOmega_n, make_gf_imtime


from triqs_xca.block_sparse_solver import BlockSparseSolver


from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators


def get_serge_florens_analytic_spgf(mesh_f_tau, U_w, verbose=False):

    """
    Analytic single particle Green's function for retarded interacting AIM

    pp. 156 
    Coherence et localisation dans les systemes d'electrons fortement correles 
    PhD thesis by Serge Florens (2003)

    """

    U_w_mat = U_w
    U_w = Gf(mesh=U_w_mat.mesh, target_shape=[])
    U_w.data[:] = U_w_mat.data[:, 0, 0].copy()

    from triqs.gfs import MatsubaraFreq

    beta = U_w.mesh.beta
    U_dlr = make_gf_dlr(U_w)
    U_0 = U_dlr(MatsubaraFreq(0, beta, 'Boson')) # zeroth Matsubara frequency component of the retarded interaction

    F_w = U_w.copy()
    for iwn in U_w.mesh:
        F_w[iwn] = (U_w[iwn] - U_0) / complex(iwn)**2 if iwn.index != 0 else 0.0

    F_tau = make_gf_dlr_imtime(F_w)

    F_dlr = make_gf_dlr(F_tau)
    F_tau -= F_dlr(0.) # subtract the tau = 0^+ value of F(\tau)
    F_dlr = make_gf_dlr(F_tau)

    beta = mesh_f_tau.beta

    G_tau = Gf(mesh=mesh_f_tau, target_shape=[1, 1])
    for tau in mesh_f_tau:
        G_tau[tau] = -0.5 * np.exp(F_dlr(tau)) * (np.exp(-U_0/2 * tau) + np.exp(-U_0/2 * (beta - tau))) / \
            (1 + np.exp(-beta * U_0/2)) 

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt

        plt.figure(figsize=(6, 8))
        subp = [3, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplot(U_w)
        plt.plot(0, U_0.real, 'ro', label='Re[U_0]')
        plt.plot(0, U_0.imag, 'bs', label='Im[U_0]')

        #plt.plot(iwn.imag, U_w_ref.real, 'r+', label='Re[U_w]')
        #plt.plot(iwn.imag, U_w_ref.imag, 'b+', label='Im[U_w]')

        #plt.plot(iwn.imag, U_w_ref2.real, 'r.', label='Re[U_w]')
        #plt.plot(iwn.imag, U_w_ref2.imag, 'b.', label='Im[U_w]')

        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_dlr_imtime(U_w), label='U_tau')

        plt.subplot(*subp); subp[-1] += 1
        oplot(F_w)

        plt.subplot(*subp); subp[-1] += 1
        oplot(F_tau, label='F_tau')

        plt.subplot(*subp); subp[-1] += 1
        oplot(G_tau, label='G_tau')

        plt.tight_layout()
        plt.show(); exit()

    return G_tau


def get_dimer_ed_ref(eps0, eps1, V, g, omega0, mesh_f_tau, mesh_b_tau, Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=2, Nb=1, Nb_max=Nb_max)
    
    c0, c1, b = ops.c_dag[0].getH(), ops.c_dag[1].getH(), ops.b_dag[0].getH()
    n0, n1, nb = c0.getH() * c0, c1.getH() * c1, b.getH() * b

    H = eps0 * n0 + eps1 * n1  + V*(c1.getH() * c0 + c0.getH() * c1) + g*(b + b.getH())*n0 + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_f_tau.beta)

    tau_f = np.array([float(t) for t in mesh_f_tau])
    g_tau = Gf(mesh=mesh_f_tau, target_shape=[1, 1])
    g_tau.data[:, 0, 0] = ed.get_tau_greens_function_component(tau_f, c0, c0.getH())

    tau_b = np.array([float(t) for t in mesh_b_tau])
    chi_tau = Gf(mesh=mesh_b_tau, target_shape=[1, 1])
    chi_tau.data[:, 0, 0] = -ed.get_tau_greens_function_component(tau_b, n0, n0)
    
    return g_tau, chi_tau


def get_ed_ref(eps0, g, omega0, mesh_f_tau, mesh_b_tau, Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=1, Nb=1, Nb_max=Nb_max)
    
    c, b = ops.c_dag[0].getH(), ops.b_dag[0].getH()
    nf, nb = c.getH() * c, b.getH() * b

    H = eps0 * nf + g*(b + b.getH())*nf + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_f_tau.beta)

    tau_f = np.array([float(t) for t in mesh_f_tau])
    g_tau = Gf(mesh=mesh_f_tau, target_shape=[1, 1])
    g_tau.data[:, 0, 0] = ed.get_tau_greens_function_component(tau_f, c, c.getH())

    tau_b = np.array([float(t) for t in mesh_b_tau])
    chi_tau = Gf(mesh=mesh_b_tau, target_shape=[1, 1])
    chi_tau.data[:, 0, 0] = -ed.get_tau_greens_function_component(tau_b, nf, nf)
    
    return g_tau, chi_tau


def test_dynint_one_fermion_1st_order(beta=2.1, eps0=-0.1, g=0.1, omega0=1., w_max=2.0, eps=1e-12, 
                                      order=1, verbose=False):
    
    """" Solve AIM with single fermionic level coupled to a bosonic mode 
    with linear coupling and retarded interaction given by the bosonic propagator. 
    Compare to ED reference solution. 
    
    Note that the retarded interaction does not contribute to the 
    single particle Green's function diagrams (they are zero for all orders).

    Thus, this only tests the pseudo particle self-energy diagrams.
    """

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    from triqs.operators import n

    S = BlockSparseSolver(
        H_loc=(eps0 - mu) * n('0', 0), 
        beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 1]], conserved_operators=[])
    
    S.Delta_tau['0'].data[:] = 0.

    f_mesh = S.mesh_tau
    b_mesh = MeshDLRImTime(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max)

    D0_tau = Gf(mesh=b_mesh, target_shape=[1, 1])
    D0_iw = make_gf_dlr_imfreq(D0_tau)
    D0_iw << -2 * g**2 * omega0 * inverse(omega0**2 - iOmega_n*iOmega_n)

    D0_iw << 0.5 * D0_iw # FIXME! Compensate for double number of Sigma diagrams for retarded interactions

    D0_tau << make_gf_dlr_imtime(D0_iw)
    D0_c = make_gf_dlr(D0_iw)

    r = S.Delta_tau['0'].data.shape[0]

    dynint_ops = [n('0', 0)]
    dynint_coeffs = np.zeros([r, 1, 1], dtype=complex)

    dynint_coeffs[:] = D0_c.data

    S.set_dynamic_interactions(dynint_ops, dynint_coeffs)
    S.solve(max_order=order, spgf_max_order=1, maxiter=8, tol=1e-8, verbose=True, hyb_comp=False)

    g_tau_ed_0, chi_tau_ed_0 = get_ed_ref(eps0 - mu, 0.0, omega0, f_mesh, b_mesh, Nb_max=10)
    g_tau_ed, chi_tau_ed = get_ed_ref(eps0 - mu, g, omega0, f_mesh, b_mesh, Nb_max=10)

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt

        plt.figure(figsize=(6, 6))
        subp = [1, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_imtime(S.G_tau, n_tau=100).real, '-', label='xca')
        oplot(make_gf_imtime(g_tau_ed, n_tau=100).real, ':', label='ed')
        oplot(make_gf_imtime(g_tau_ed_0, n_tau=100).real, ':', label='ed (g=0)')
        plt.show()

    error = np.max(np.abs(S.G_tau['0'].data - g_tau_ed.data))
    return error


def test_dynint_retarded_dimer():

    beta = 2.

    eps0 = -0.1
    g = 0.1
    omega0 = 1.

    #eps0 = 0.0
    #g = 1.0
    #omega0 = 1.

    #eps0, eps1, V, g, omega0 = -0.1, +0.1, 0., 0.1, 1. # Debug V = 0.0
    #eps0, eps1, V, g, omega0 = -0.1, +0.1, 0., 0., 1. # Debug g = 0.0, V = 0.0
    #eps0, eps1, V, g, omega0 = -0.1, +0.1, 0.25, 0., 1. # DEBUG g = 0.0
    #eps0, eps1, V, g, omega0 = -0.1, +0.1, 0.25, 1., 1. # Original

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    from triqs.operators import n, c, c_dag

    H = (eps0 - mu) * n('0', 0)
    #H = (eps0 - mu) * n('0', 0) + eps1 * n('0', 1) + V * (c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0))

    S = BlockSparseSolver(
        #H, beta, w_max=4.0, eps=1e-10, gf_struct=[['0', 2]], conserved_operators=[])
        H, beta, w_max=2.0, eps=1e-12, gf_struct=[['0', 1]], conserved_operators=[])
    
    S.Delta_tau['0'].data[:] = 0.

    f_mesh = S.mesh_tau
    b_mesh = MeshDLRImTime(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max)

    D0_tau = Gf(mesh=b_mesh, target_shape=[1, 1])
    D0_iw = make_gf_dlr_imfreq(D0_tau)
    D0_iw << -2 * g**2 * omega0 * inverse(omega0**2 - iOmega_n*iOmega_n)

    D0_iw << 0.5 * D0_iw # FIXME! Compensate for double number of Sigma diagrams for retarded interactions

    D0_tau << make_gf_dlr_imtime(D0_iw)
    D0_c = make_gf_dlr(D0_iw)

    r = S.Delta_tau['0'].data.shape[0]

    dynint_ops = [n('0', 0)]
    dynint_coeffs = np.zeros([r, 1, 1], dtype=complex)

    dynint_coeffs[:] = D0_c.data
    #dynint_coeffs[:] = 0. # Debug: check that we recover the non-interacting limit when the dynamic interaction is turned off
    print(f'Dynamic interaction coefficients:\n{dynint_coeffs.flatten()}')

    # Check agreement if the mesh is fermionic
    D0_tau_f = Gf(mesh=f_mesh, target_shape=[1, 1])

    D0_dlr_f = make_gf_dlr(D0_tau_f)
    D0_dlr_f.data[:] = D0_c.data

    for tau in f_mesh:
        D0_tau_f[tau] = D0_dlr_f(float(tau))

    np.testing.assert_array_almost_equal(D0_tau_f.data, D0_tau.data)

    S.set_dynamic_interactions(dynint_ops, dynint_coeffs)

    S.solve(max_order=1, spgf_max_order=1, maxiter=1, tol=1e-8, verbose=True, hyb_comp=False)

    Sigma1 = S._BlockSparseSolver__eval_pseudo_particle_self_energy_order(S.G0, order=1, connected=True)
    Sigma2 = S._BlockSparseSolver__eval_pseudo_particle_self_energy_order(S.G0, order=2, connected=True)
    Sigma3 = S._BlockSparseSolver__eval_pseudo_particle_self_energy_order(S.G0, order=3, connected=True)

    # -- Loop over topologies and indices.

    def eval_pseudo_particle_self_energy_topology_loop(self, G, topology, verbose=False):

        order = len(topology)
        n_max = self.d.get_num_self_energy_backbones(topology)

        from triqs_xca.block_sparse_solver import scatter_array_over_ranks
        n_vec = scatter_array_over_ranks(np.arange(n_max, dtype=np.int32))

        Sigmas = []
        for n in n_vec:
             small_n_vec = np.array([n], dtype=np.int32)
             Sigma = self.get_zero_pseudo_particle_propagator()
             
             Sigma = pow(-1, order+1) * self.d.compute_self_energy(G, topology, small_n_vec)
             Sigmas.append(Sigma)

        return Sigmas

    topology = [(0, 1)]
    topology = np.array(topology, dtype=np.int32)

    Sigmas = eval_pseudo_particle_self_energy_topology_loop(S, S.G, topology, verbose=True)
    #exit()

    g_tau_ref = S.eval_one_time_correlator(S.G, max_order=1, ops_tau=[c('0', 0)], ops_0=[c_dag('0', 0)])
    chi_tau = -S.eval_one_time_correlator(S.G, max_order=1, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    #print(f'f_mesh = \n{f_mesh}')
    #print(f'b_mesh = \n{b_mesh}')

    g_tau_ed_0, chi_tau_ed_0 = get_ed_ref(eps0 - mu, 0.0, omega0, f_mesh, b_mesh, Nb_max=10)
    g_tau_ed, chi_tau_ed = get_ed_ref(eps0 - mu, g, omega0, f_mesh, b_mesh, Nb_max=10)

    g_tau_anal = get_serge_florens_analytic_spgf(f_mesh, D0_iw)

    from triqs.plot.mpl_interface import oplot, plt

    plt.figure(figsize=(6, 8))
    subp = [4, 2, 1]

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(S.G, n_tau=100), '-', label=None)
    oplot(make_gf_imtime(S.G0, n_tau=100), ':', label=None)
    plt.plot([], [], '-', color='gray', label='G')
    plt.plot([], [], ':', color='gray', label='G0')
    plt.legend(loc='best')
    plt.ylabel(r'$G(\tau)$, $G_0(\tau)$')

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(Sigma1, n_tau=100), '--', label=None)
    oplot(make_gf_imtime(Sigma2, n_tau=100), '-.', label=None)
    oplot(make_gf_imtime(Sigma3, n_tau=100), ':', label=None)

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(S.Sigma, n_tau=100), '-', label=None)
    oplot(make_gf_imtime(Sigma1, n_tau=100), '--', label=None)
    oplot(make_gf_imtime(Sigma2, n_tau=100), '-.', label=None)
    plt.ylabel(r'$\Sigma(\tau)$')

    for i, Sigma in enumerate(Sigmas):
        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_imtime(Sigma, n_tau=100), '--', label=None)
        plt.ylabel(r'$\Sigma(\tau)$' + f' f_ix={i}')

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(D0_tau, n_tau=100).real, '-x', label='D0_tau')
    oplot(make_gf_imtime(D0_tau_f, n_tau=100).real, '--+', label='D0_tau_f')

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(S.G_tau['0'][0, 0], n_tau=100).real, '-', label='g_tau_xca')
    oplot(make_gf_imtime(g_tau_anal, n_tau=100).real, '-.', label='g_tau_anal')
    oplot(make_gf_imtime(g_tau_ref, n_tau=100).real, '-.', label='g_tau_xca_ref')
    oplot(make_gf_imtime(g_tau_ed, n_tau=100).real, '--', label='g_tau_ed')
    oplot(make_gf_imtime(g_tau_ed_0, n_tau=100).real, ':', label='g_tau_ed_0')

    plt.subplot(*subp); subp[-1] += 1
    oplot(make_gf_imtime(chi_tau, n_tau=100).real, '-', label='chi_tau_xca')
    oplot(make_gf_imtime(chi_tau_ed, n_tau=100).real, '--', label='chi_tau_ed')
    oplot(make_gf_imtime(chi_tau_ed_0, n_tau=100).real, ':', label='chi_tau_ed_0')

    plt.tight_layout()
    plt.show()


def test_dynint_hubbard_atom():

    U = 1.0
    mu = 0.5 * U
    beta = 1.33

    from triqs.operators import n
    
    H = -mu * (n('0', 0) + n('0', 1)) + U * n('0', 0) * n('0', 1)

    S = BlockSparseSolver(
        H, beta, w_max=4.0, eps=1e-10, gf_struct=[['0', 2]], conserved_operators=[])

    S.Delta_tau['0'].data[:] = 0.

    r = S.Delta_tau['0'].data.shape[0]

    dynint_ops = [n('0', 0), n('0', 1)]
    dynint_coeffs = np.zeros([r, 2, 2], dtype=complex)

    S.set_dynamic_interactions(dynint_ops, dynint_coeffs)

    S.solve(max_order=1, tol=1e-8, verbose=True)


def test_convergence_rate(verbose=False):

    errss = []
    orders = [1, 2, 3]
    for order in orders:
        g2s = np.logspace(-1.5, -0.5, 3)
        errs = np.zeros_like(g2s)
        for i, g2 in enumerate(g2s):
            g = np.sqrt(g2)
            errs[i] = test_dynint_one_fermion_1st_order(g=g, order=order)
        errss.append(errs)

    if mpi.is_master_node():
        # Compute convergence rates
        rates = []
        for order, errs in zip(orders, errss):
            rate = (np.log(errs[:-1] / errs[1:]) / np.log(g2s[:-1] / g2s[1:]))[0]
            rates.append(rate)
            print(f'Order {order} convergence rates: {rate}')

    if verbose and mpi.is_master_node():
        import matplotlib.pyplot as plt
        for i, (order, errs) in enumerate(zip(orders, errss)):
            plt.loglog(g2s, errs, 'o-', label=f'O{order}')
        plt.xlabel('$g^2$')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    if mpi.is_master_node():
        # Test convergence rates
        for order, rate in zip(orders, rates):
            diff = np.abs(rate - (order + 1))
            assert( diff < 0.2 ), f'Expected convergence rate of {order+1} for order {order}, but got {rate}, diff {diff}'


def is_permutation(p):
    """Check if p is a permutation of 0..n-1."""
    return sorted(p) == list(range(len(p)))


def permutation_parity(p):
    """Return the parity of a permutation of 0..n.

    Returns:
        +1 for an even permutation, -1 for an odd permutation.
    """
    if not is_permutation(p):
        raise ValueError("p must be a permutation of integers 0..len(p)-1")

    n = len(p)
    visited = [False] * n
    parity = 0

    # Each cycle of length L contributes L-1 transpositions.
    for i in range(n):
        if visited[i]:
            continue

        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = p[j] - 1
            cycle_len += 1

        parity ^= (cycle_len - 1) & 1

    return +1 if parity == 0 else -1


def remove_permutation_element(p, idx):

    e = p.pop(idx)
    for i in range(len(p)):
        if p[i] > e:
            p[i] -= 1

    return p, e


def remove_topology_pair(topology, idx):

    print(f'--> Removing pair at index {idx} from topology {topology}')
    pair = topology[idx]
    print(f'    Pair to remove: {pair}')

    perm = permutation_from_topology(topology)
    start_sign = -permutation_parity(perm)
    print(f'    Initial permutation: {perm} with parity {start_sign:+d}')

    perm, e1 = remove_permutation_element(perm, 2 * idx)
    print(f'    After removing element {e1}: permutation {perm}')
    perm, e2 = remove_permutation_element(perm, 2 * idx)
    print(f'    After removing element {e2}: permutation {perm}')

    assert( e1 == pair[0] and e2 + 1 == pair[1] ), f'Expected to remove pair {pair} but got {e1}, {e2}'

    delta_sign = (-1)**(pair[1] - pair[0] - 1)
    end_sign = -permutation_parity(perm)
    topology = topology_from_permutation(perm)

    assert( start_sign * delta_sign == end_sign ), \
        f'Sign mismatch: start {start_sign}, delta {delta_sign}, end {end_sign}'
    
    return topology, delta_sign


def permutation_from_topology(topology):
    perm = np.array(topology, dtype=int).flatten().tolist()
    return perm


def topology_from_permutation(perm):

    assert( len(perm) % 2 == 0 ), f'Permutation length must be even, got {len(perm)}'

    topology = np.array(perm, dtype=int).reshape(-1, 2).tolist()
    topology = [tuple(pair) for pair in topology]

    return topology


def analyze_signs(connected=False):

    from triqs_xca.diag import all_pairings, all_connected_pairings
    pairings = all_connected_pairings if connected else all_pairings

    orders = np.arange(1, 3, 1)

    for order in orders:
        for sign, topology in pairings(order):

            perm = permutation_from_topology(topology)
            sign_ref = -permutation_parity(perm)

            print(f'sign {sign:+d} sign_ref {sign_ref:+d} topo {topology} perm {perm}')

            assert( sign == sign_ref ), f'Sign mismatch for topology {topology}: expected {sign_ref}, got {sign}'

            #top, dsign = remove_topology_pair(topology, 0)
            #print(f'  After removing pair {topology[0]}: topo {top} dsign {dsign:+d}')


if __name__ == '__main__':

    #test_dynint_hubbard_atom()
    test_dynint_retarded_dimer()
    #test_dynint_one_fermion_1st_order(verbose=True)
    #test_convergence_rate(verbose=True)
    #analyze_signs()