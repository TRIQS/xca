
import numpy as np

from h5 import HDFArchive

from triqs.operators.util.U_matrix import U_matrix_slater
from triqs.operators.util.hamiltonians import h_int_slater

from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori


from triqs_xca.block_sparse_solver import is_root


def solve_slater_condon_bethe_half_filling(
        l=2, # d-orbitals angular momentum quantum number
        Fs=[3.0, 0.5, 0.3], # Slater-Condon interaction parameters for d-orbitals
        beta=1.0,
        order=1, 
        eps=1e-3,
        ppsc_tol=1e-2,
        ppsc_maxiter=10,
        ):

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
        raise NotImplementedError('Only l=2 (d-orbitals) and l=1 (p-orbitals) are implemented for now.')

    # --

    n_orb = 2*l + 1
    spin_names = ['up', 'do']
    gf_struct = [ [f'{spin}', n_orb] for spin in spin_names ]

    U_matrix = U_matrix_slater(l, radial_integrals=Fs)
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
        #conserved_operators=[N_tot],
        )

    S.Delta_tau['up'] << Delta_tau
    S.Delta_tau['do'] << Delta_tau

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

if __name__ == '__main__':

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

