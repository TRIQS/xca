

import numpy as np

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, MeshDLRImFreq, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_dlr, inverse, iOmega_n, make_gf_imtime


from triqs_xca import Solver

from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators


from common import plot_comparison, test_convergence_rate, Dummy


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


def pyed_dimer_dynint(mesh_tau, eps0=-0.1, eps1=0.1, V=0.5, g=0.0, omega0=1., Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=2, Nb=1, Nb_max=Nb_max)
    
    c0, c1, b = ops.c_dag[0].getH(), ops.c_dag[1].getH(), ops.b_dag[0].getH()
    n0, n1, nb = c0.getH() * c0, c1.getH() * c1, b.getH() * b

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    H = (eps0 - mu) * n0 + eps1 * n1  + V*(c1.getH() * c0 + c0.getH() * c1) + g*(b + b.getH())*n0 + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_tau.beta)

    tau_f = np.array([float(t) for t in mesh_tau])
    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    G_tau.data[:] = ed.get_tau_greens_function_component(tau_f, c0, c0.getH())

    tau_b = np.array([float(t) for t in mesh_tau])
    Chi_tau = Gf(mesh=mesh_tau, target_shape=[])
    Chi_tau.data[:] = ed.get_tau_greens_function_component(tau_b, n0, n0)
    
    d = Dummy()
    d.G_tau = G_tau
    d.Chi_tau = Chi_tau
    return d


def pyed_one_fermion_dynint(mesh_tau, eps0=-0.1, g=0.0, omega0=1., Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=1, Nb=1, Nb_max=Nb_max)
    
    c, b = ops.c_dag[0].getH(), ops.b_dag[0].getH()
    nf, nb = c.getH() * c, b.getH() * b

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    H = (eps0 - mu) * nf + g*(b + b.getH())*nf + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_tau.beta)

    tau_f = np.array([float(t) for t in mesh_tau])
    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    G_tau.data[:] = ed.get_tau_greens_function_component(tau_f, c, c.getH())

    tau_b = np.array([float(t) for t in mesh_tau])
    Chi_tau = Gf(mesh=mesh_tau, target_shape=[])
    Chi_tau.data[:] = ed.get_tau_greens_function_component(tau_b, nf, nf)
    
    d = Dummy()
    d.G_tau = G_tau
    d.Chi_tau = Chi_tau
    return d


def xca_dimer_dynint(
        mesh_tau,
        eps0=-0.1, eps1=0.1, V=0.5, g=0.0, omega0=1.,
        sigma_order=1, spgf_order=None, verbose=False):
    
    """" Solve AIM with single fermionic level coupled to a bosonic mode 
    with linear coupling and retarded interaction given by the bosonic propagator. 
    Compare to ED reference solution. 
    
    Note that the retarded interaction does not contribute to the 
    single particle Green's function diagrams (they are zero for all orders).

    Thus, this only tests the pseudo particle self-energy diagrams.
    """

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    from triqs.operators import n

    S = Solver(
        H_loc=(eps0 - mu) * n('0', 0), gf_struct=[['0', 1]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=[])
    
    Delta_w = make_gf_dlr_imfreq(S.Delta_tau['0'])
    Delta_w << V**2 * inverse(iOmega_n - eps1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    S.Delta_tau['0'] = Delta_tau

    f_mesh = S.mesh_tau
    #b_mesh = MeshDLRImTime(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max)
    wb_mesh = MeshDLRImFreq(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max)

    D0_iw = Gf(mesh=wb_mesh, target_shape=[1, 1])
    D0_iw << -2 * g**2 * omega0 * inverse(omega0**2 - iOmega_n*iOmega_n)
    D0_iw << 0.5 * D0_iw # FIXME! Compensate for double number of Sigma diagrams for retarded interactions
    D0_tau = make_gf_dlr_imtime(D0_iw)

    S.set_dynamic_interactions(dynint_ops=[n('0', 0)], dynint_tau=D0_tau)

    S.solve(max_order=sigma_order, spgf_max_order=spgf_order, maxiter=8, tol=1e-10, verbose=True, hyb_comp=True, hyb_tol=1e-10)

    S.Chi_tau = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    d = Dummy()
    d.S = S
    d.sigma_order = sigma_order
    d.spgf_order = spgf_order
    d.G_tau = S.G_tau['0'][0, 0].copy()
    d.Chi_tau = S.Chi_tau[0, 0].copy()

    return d


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

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-10, w_max=2.0, symmetrize=False)

    def pyed_dimer_dynint_V(mesh_tau, t=1.0):
        return pyed_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=t, g=0.0, omega0=1., Nb_max=10)

    def xca_dimer_dynint_V(mesh_tau, t=1.0, sigma_order=1, spgf_order=None, verbose=False):
        return xca_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=t, g=0.0, omega0=1.,
            sigma_order=sigma_order, spgf_order=spgf_order, verbose=verbose)


    def pyed_dimer_dynint_g(mesh_tau, t=1.0):
        return pyed_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=0., g=t, omega0=1., Nb_max=10)

    def xca_dimer_dynint_g(mesh_tau, t=1.0, sigma_order=1, spgf_order=None, verbose=False):
        return xca_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=0., g=t, omega0=1., 
            sigma_order=sigma_order, spgf_order=spgf_order, verbose=verbose)


    def pyed_dimer_dynint_Vg(mesh_tau, t=1.0):
        return pyed_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=t, g=t, omega0=1., Nb_max=10)

    def xca_dimer_dynint_Vg(mesh_tau, t=1.0, sigma_order=1, spgf_order=None, verbose=False):
        return xca_dimer_dynint(
            mesh_tau, eps0=-0.1, eps1=0.1, V=t, g=t, omega0=1., 
            sigma_order=sigma_order, spgf_order=spgf_order, verbose=verbose)


    #plot_comparison(m_dlr, pyed_dimer_dynint_V, xca_dimer_dynint_V, max_order=3, t=0.1)
    #plot_comparison(m_dlr, pyed_dimer_dynint_g, xca_dimer_dynint_g, max_order=3, t=0.5)
    #plot_comparison(m_dlr, pyed_dimer_dynint_Vg, xca_dimer_dynint_Vg, max_order=3, t=0.5)

    #test_convergence_rate(m_dlr, pyed_dimer_dynint_V, xca_dimer_dynint_V, label='dimer_dynint_V', max_order=4)
    #test_convergence_rate(m_dlr, pyed_dimer_dynint_g, xca_dimer_dynint_g, label='dimer_dynint_g', max_order=3)
    test_convergence_rate(m_dlr, pyed_dimer_dynint_Vg, xca_dimer_dynint_Vg, label='dimer_dynint_Vg', max_order=3)

    #analyze_signs()