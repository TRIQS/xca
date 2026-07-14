
import numpy as np

from h5 import HDFArchive

from triqs.operators.util.U_matrix import U_matrix_slater
from triqs.operators.util.hamiltonians import h_int_slater

from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori

from triqs.gf import Gf
from triqs.operators import n

from triqs_xca.block_sparse_solver import is_root

from adapol.fit_utils_dlr import polefitting_dlr_triqs

from mpi4py import MPI
from time import perf_counter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def _N_tot_operator(l):
    """Total particle number operator for the (l-shell, spin up/down) orbital content."""

    n_orb = 2*l + 1
    spin_names = ['up', 'do']
    return sum(n(spin, oidx) for spin in spin_names for oidx in range(n_orb))


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

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps, symmetrize=False)
    Delta_w = Gf(mesh=mesh_w, target_shape=[n_orb]*2)
    Delta_w << SemiCircular(half_bandwidth=1.0)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    return H, N_tot, gf_struct, Delta_tau, w_max


def _from_blockgf_to_dense(G):
    """Convert a BlockGf to a dense Gf.
    
    Parameters
    ----------
    G : BlockGf
        Block Green's function to be converted.
        
    Returns
    -------
    G_dense : Gf
        Dense Green's function with shape [sum(block_sizes), sum(block_sizes)].
    """
    for b, g in G:
        assert( len(g.target_shape) == 2)
        assert( g.target_shape[0] == g.target_shape[1] )

    norb = sum([ g.target_shape[0] for b, g in G ])
    
    G_dense = Gf(mesh=G.mesh, target_shape=[norb]*2)
    
    sidx = 0
    for b, g in G:
        size = g.target_shape[0]
        G_dense.data[:, sidx:sidx+size, sidx:sidx+size] = g.data
        sidx += size

    return G_dense


def _build_slater_condon_solver(l, Fs, beta, eps, conserved_operators):

    H, N_tot, gf_struct, Delta_tau, w_max = _setup_slater_condon_problem(l, Fs, beta, eps)

    from triqs_xca.block_sparse_solver import BlockSparseSolver

    S = BlockSparseSolver(
        H, beta, w_max, eps, gf_struct,
        conserved_operators=conserved_operators,
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


def _prepare_dense_triqs_solver(S, H, verbose=False):
    """Initialize TriqsSolver internals without running the expensive SPGF evaluation."""

    S.order = None
    S.h_int = H

    S.S.set_H_loc(H)
    S.S.G_iaa = S.S.G0_iaa.copy()

    delta_iaa = S._TriqsSolver__from_blockgf_to_array(S.Delta_tau)
    S.S.set_hybridization(delta_iaa, compress=True, verbose=verbose)

    return delta_iaa


def solve_slater_condon_bethe_half_filling(
        conserved_operators,
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ppsc_maxiter=10,
        ):

    S, N_tot = _build_slater_condon_solver(l=l, Fs=Fs, beta=beta, eps=eps, conserved_operators=conserved_operators)

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
        conserved_operators,
        dense=False,
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ):

    section_times = {}

    t_section = perf_counter()
    if dense:
        S, H, N_tot = _build_slater_condon_solver_dense(l=l, Fs=Fs, beta=beta, eps=eps)
    else:
        S, N_tot = _build_slater_condon_solver(l=l, Fs=Fs, beta=beta, eps=eps, conserved_operators=conserved_operators)
    section_times['solver_setup'] = perf_counter() - t_section

    # compute hybridization poles and weights using adapol
    # use same poles and weights for dense and block-sparse solvers
    # copied over from block_sparse_solver.py fit_hybridization() method
    t_section = perf_counter()
    tol_adapol = ppsc_tol
    Delta_tau_dense = _from_blockgf_to_dense(S.Delta_tau)
    weights, poles, _ = polefitting_dlr_triqs(Delta_tau_dense, eps=tol_adapol, statistics="Fermion", verbose=True)
    weights = -1 * weights
    # print(f'Weights = {weights}')
    # print(f'Poles = {poles}')
    section_times['hybridization_fit'] = perf_counter() - t_section
    
    # compute the self-energy at the given order using the non-interacting G 
    if dense:
        t_section = perf_counter()
        _prepare_dense_triqs_solver(S, H)
        S.S.fd.copy_aaa_result(poles, weights)
        S.S.fd.hyb_decomposition(poledlrflag=False, eps=0.0)
        section_times['dense_solver_prep'] = perf_counter() - t_section

        from triqs_xca.diag import all_connected_pairings

        Sigma_dense = 0 # zeros like S.G_iaa
        # comm.Barrier()
        t_start = MPI.Wtime()
        t_section = perf_counter()
        for sign, topo in all_connected_pairings(order):
            Sigma_dense += pow(-1, order) * sign * S.S.calc_Sigma_topology(topo)
        section_times['dense_eval'] = perf_counter() - t_section
        # comm.Barrier()
        t_end = MPI.Wtime()
        elapsed = section_times['dense_eval']
    else:
        t_section = perf_counter()
        S.set_hybridization_poles_and_coefficients(poles, weights)
        S.init_diagram_evaluator()
        section_times['block_sparse_solver_prep'] = perf_counter() - t_section

        # comm.Barrier()
        t_start = MPI.Wtime()
        t_section = perf_counter()
        Sigma = S._BlockSparseSolver__eval_pseudo_particle_self_energy_order(S.G, order, connected=True)
        section_times['block_sparse_eval'] = perf_counter() - t_section
        # comm.Barrier()
        t_end = MPI.Wtime()
        elapsed = section_times['block_sparse_eval']
        from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense
        t_section = perf_counter()
        Sigma_dense = pseudo_particle_block_gf_to_dense(Sigma, S.atom_diag)
        section_times['block_sparse_to_dense'] = perf_counter() - t_section

    # elapsed = t_end - t_start

    if is_root():
        filename = f"{'dense' if dense else 'bs'}_self_energy_l_{l}_order_{order}_beta_{S.beta}.h5"
        output_filenames = [filename]
        # [PROFILING ADDITION] For MPI runs, also write per-rank HDF5 outputs to profile individual processes
        if comm.Get_size() > 1:
            output_filenames.append(filename.replace('.h5', f'.rank{rank}.h5'))

        # [PROFILING ADDITION] Write to both root summary and per-rank profile files (if MPI)
        for output_filename in output_filenames:
            with HDFArchive(output_filename, 'w') as ar:
                ar['l'] = l
                ar['order'] = order
                # [PROFILING ADDITION] Store rank metadata for MPI profiling
                ar['rank'] = rank
                ar['comm_size'] = comm.Get_size()
                ar['Sigma_dense'] = Sigma_dense
                if not dense:
                    ar['Sigma'] = Sigma
                ar['elapsed_time'] = elapsed
                ar['section_times'] = section_times

        print('\nDetailed section timings:')
        for key, value in section_times.items():
            print(f'  {key}: {value:.6f} s')

    return Sigma_dense

def one_spgf_iter_slater_condon_bethe_half_filling(
        conserved_operators,
        dense=False,
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ):

    # build solver first to get S and Delta_tau
    if dense:
        S, H, N_tot = _build_slater_condon_solver_dense(l=l, Fs=Fs, beta=beta, eps=eps)
    else:
        S, N_tot = _build_slater_condon_solver(l=l, Fs=Fs, beta=beta, eps=eps, conserved_operators=conserved_operators)
    
    # compute hybridization poles and weights using adapol
    # use same poles and weights for dense and block-sparse solvers
    # copied over from block_sparse_solver.py fit_hybridization() method
    tol_adapol = ppsc_tol
    Delta_tau_dense = _from_blockgf_to_dense(S.Delta_tau)
    weights, poles, _ = polefitting_dlr_triqs(Delta_tau_dense, eps=tol_adapol, statistics="Fermion", verbose=True)
    weights = -1 * weights

    if dense:
        _prepare_dense_triqs_solver(S, H)
        S.S.fd.copy_aaa_result(poles, weights)
        S.S.fd.hyb_decomposition(poledlrflag=False, eps=0.0)

        from triqs_xca.diag import all_connected_pairings

        spgf = 0
        # comm.Barrier()
        t_start = MPI.Wtime()
        for sign, topo in all_connected_pairings(order):
            spgf += pow(-1, order) * sign * S.S.calc_spgf_toplogy(topo)
        # comm.Barrier()
        t_end = MPI.Wtime()
    else:
        S.set_hybridization_poles_and_coefficients(poles, weights)
        S.init_diagram_evaluator()

        # comm.Barrier()
        t_start = MPI.Wtime()
        spgf = S._BlockSparseSolver__eval_single_particle_greens_function_order(S.G, order)
        # comm.Barrier()
        t_end = MPI.Wtime()

    elapsed = t_end - t_start

    if is_root():
        filename = f'{'dense' if dense else 'bs'}_spgf_l_{l}_order_{order}_beta_{S.beta}.h5'
        with HDFArchive(filename, 'w') as ar:
            ar['l'] = l
            ar['order'] = order
            ar['G'] = spgf
            ar['elapsed_time'] = elapsed

    return spgf

if __name__ == '__main__':

    run_full_solves = False
    run_one_se_no_sym_iters = False
    run_one_se_all_sym_iters = True
    run_one_spgf_no_sym_iters = False

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

    if run_one_se_no_sym_iters:
        orders = [1]#, 2, 3]

        for order in orders:
            for dense in [False, True]:
                opts = dict(
                    dense=dense,
                    beta=1.0,
                    eps=1e-9,
                    ppsc_tol=1e-4,
                    order=order,
                )

                one_se_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=0, Fs=[3.0], **opts)
                one_se_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=1, Fs=[3.0, 0.5], **opts)
                # one_se_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=2, Fs=[3.0, 0.5, 0.3], **opts)
    
    if run_one_se_all_sym_iters:
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

                one_se_iter_slater_condon_bethe_half_filling(conserved_operators=[_N_tot_operator(0)], l=0, Fs=[3.0], **opts)
                one_se_iter_slater_condon_bethe_half_filling(conserved_operators=[_N_tot_operator(1)], l=1, Fs=[3.0, 0.5], **opts)
                # one_se_iter_slater_condon_bethe_half_filling(conserved_operators='automatic', l=2, Fs=[3.0, 0.5, 0.3], **opts)

    if run_one_spgf_no_sym_iters:
        orders = [2] # [1, 2, 3]

        for order in orders:
            for dense in [False, True]:
                opts = dict(
                    dense=dense,
                    beta=1.0,
                    eps=1e-9,
                    ppsc_tol=1e-4,
                    order=order,
                )

                one_spgf_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=0, Fs=[3.0], **opts)
                # one_spgf_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=1, Fs=[3.0, 0.5], **opts)
                # one_spgf_iter_slater_condon_bethe_half_filling(conserved_operators=[], l=2, Fs=[3.0, 0.5, 0.3], **opts)
