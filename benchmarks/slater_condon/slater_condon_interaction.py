
import numpy as np

from h5 import HDFArchive

from triqs.operators.util.U_matrix import U_matrix_slater
from triqs.operators.util.hamiltonians import h_int_slater

from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori


from triqs_xca.block_sparse_solver import is_root


def _slater_condon_mu_wmax(l, Fs):
    """Return half-filling chemical potential and recommended DLR cutoff."""

    if l == 2:
        assert( len(Fs) == 3 )
        F0, F2, F4 = Fs
        mu = (45*F0 - 70/49*F2 - 630/441*F4) / 10. # For half-filling of the d-shell
        w_max = 75.0 * F0 # 10 electrons with energy scale 75 * F0
    elif l == 1:
        assert( len(Fs) == 2 )
        F0, F2 = Fs
        mu = (15 * F0 - 6/5 * F2) / 6 # For half-filling of the p-shell
        w_max = 15.0 * F0 # 6 electrons with energy scale 15 * F0
    elif l == 0:
        assert( len(Fs) == 1 )
        F0, = Fs
        mu = F0 / 2. # For half-filling of the s-shell
        w_max = F0 # 2 electrons with energy scale F0
    else:
        raise NotImplementedError('Only l=0 (s-orbitals), l=1 (p-orbitals), and l=2 (d-orbitals) are implemented for now.')

    return mu, w_max


def _setup_slater_condon_problem(l, Fs, beta, eps):
    """Build shared Slater-Condon inputs for both solver frontends."""

    mu, w_max = _slater_condon_mu_wmax(l, Fs)

    n_orb = 2*l + 1
    spin_names = ['up', 'do']
    gf_struct = [ [f'{spin}', n_orb] for spin in spin_names ]

    U_matrix = U_matrix_slater(l, radial_integrals=Fs)
    H = h_int_slater(spin_names, n_orb, U_matrix, off_diag=True)

    from triqs.operators import n
    N_tot = sum( n(spin, oidx) for spin in spin_names for oidx in range(n_orb) )
    H += - mu * N_tot

    from triqs.gfs import SemiCircular
    from triqs.gfs import MeshDLRImFreq, Gf, make_gf_dlr_imtime, iOmega_n, inverse

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[n_orb]*2)
    Delta_w << SemiCircular(half_bandwidth=1.0)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    return H, N_tot, gf_struct, Delta_tau, w_max


def _build_slater_condon_solver(l, Fs, beta, eps):

    H, N_tot, gf_struct, Delta_tau, w_max = _setup_slater_condon_problem(l, Fs, beta, eps)

    from triqs_xca.block_sparse_solver import BlockSparseSolver

    S = BlockSparseSolver(
        H, beta, w_max, eps, gf_struct,
        #conserved_operators=[N_tot],
        )

    S.Delta_tau['up'] << Delta_tau
    S.Delta_tau['do'] << Delta_tau

    return S, N_tot


def _build_slater_condon_solver_dense(l, Fs, beta, eps, verbose=True):
    """Build a Slater-Condon impurity model using TriqsSolver instead of BlockSparseSolver.

    Returns
    -------
    S : TriqsSolver
        Configured solver instance with Delta_tau initialized.
    H : Operator
        Local Hamiltonian including the half-filling chemical potential shift.
    N_tot : Operator
        Total particle number operator.
    """

    H, N_tot, gf_struct, Delta_tau, w_max = _setup_slater_condon_problem(l, Fs, beta, eps)

    from triqs_xca.triqs_solver import TriqsSolver

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max, verbose=verbose)
    # S.set_hybridization(Delta_tau.data())
    S.Delta_tau['up'] << Delta_tau
    S.Delta_tau['do'] << Delta_tau

    return S, H, N_tot


def solve_slater_condon_bethe_half_filling(
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ppsc_maxiter=10,
        ):

    S, N_tot = _build_slater_condon_solver(l=l, Fs=Fs, beta=beta, eps=eps)

    S.solve(max_order=order, tol=ppsc_tol, maxiter=ppsc_maxiter)

    S.l = l
    S.order = order
    S.N_tot_exp = S.expectation_value(N_tot)
    print(f'N_tot_exp = {S.N_tot_exp}')

    if is_root():
        filename = f'data_l_{S.l}_order_{S.order}_beta_{S.beta}.h5'
        with HDFArchive(filename, 'w') as ar:
            ar['S'] = S

    return S

def one_se_iter_slater_condon_bethe_half_filling(
        dense=False,
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ):

    # compute the self-energy at the given order using the non-interacting G 
    if dense:
        S, H, N_tot = _build_slater_condon_solver_dense(l=l, Fs=Fs, beta=beta, eps=eps)
        S.order = order
        S.h_int = H
        S.S.set_H_loc(H)
        S.S.G_iaa = S.S.G0_iaa.copy() # start with non-interacting G
        S.delta_iaa = S._TriqsSolver__from_blockgf_to_array(S.Delta_tau)
        S.S.set_hybridization(S.delta_iaa, compress=True, verbose=True)

        from triqs_xca.diag import all_connected_pairings
        from triqs_xca.solver import Sigma_calc_topology
        import time

        Sigma_dense = 0 # zeros like S.G_iaa
        t_start = time.perf_counter()
        for sign, topo in all_connected_pairings(order):
            Sigma_dense += pow(-1, order) * sign * S.S.calc_Sigma_topology(topo)
        t_end = time.perf_counter()
    else:
        S, N_tot = _build_slater_condon_solver(l=l, Fs=Fs, beta=beta, eps=eps)
        # copy over beginning of solve() method
        S.fit_hybridization(tol=ppsc_tol)
        S.init_diagram_evaluator()

        import time
        t_start = time.perf_counter()
        Sigma = S.eval_pseudo_particle_self_energy_order(S.G, order)
        t_end = time.perf_counter()
        from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense
        Sigma_dense = pseudo_particle_block_gf_to_dense(Sigma, S.ad)

    elapsed = t_end - t_start

    if is_root():
        filename = f'{'dense' if dense else 'bs'}_self_energy_l_{l}_order_{order}_beta_{S.beta}.h5'
        with HDFArchive(filename, 'w') as ar:
            ar['l'] = l
            ar['order'] = order
            ar['Sigma_dense'] = Sigma_dense
            if not dense:
                ar['Sigma'] = Sigma
            ar['elapsed_time'] = elapsed

    return Sigma_dense

if __name__ == '__main__':

    run_full_solves = False
    run_one_se_iters = True

    if run_full_solves:
        opts = dict(
            beta=1.0,
            eps=1e-9,
            ppsc_tol=1e-4,
            ppsc_maxiter=10,
            order=1,
        )

        solve_slater_condon_bethe_half_filling(l=0, Fs=[3.0], **opts)
        solve_slater_condon_bethe_half_filling(l=1, Fs=[3.0, 0.5], **opts)
        solve_slater_condon_bethe_half_filling(l=2, Fs=[3.0, 0.5, 0.3], **opts)

    if run_one_se_iters:
        orders = [1, 2, 3]

        for order in orders:
            for dense in [False, True]:
                opts = dict(
                    dense=dense,
                    beta=1.0,
                    eps=1e-9,
                    ppsc_tol=1e-4,
                    order=order,
                )

                one_se_iter_slater_condon_bethe_half_filling(l=0, Fs=[3.0], **opts)
                one_se_iter_slater_condon_bethe_half_filling(l=1, Fs=[3.0, 0.5], **opts)
                one_se_iter_slater_condon_bethe_half_filling(l=2, Fs=[3.0, 0.5, 0.3], **opts)

