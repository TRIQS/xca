"""Check the effect of different normalization schemes on the convergence 
of the block-sparse solver for a simple Slater-Condon interaction at half-filling. 
The test is performed for s-orbitals (l=0) with a single Slater-Condon parameter F0, 
which corresponds to the Hubbard U. The chemical potential is set to U/2 to ensure half-filling. 

The solver is run for different normalization schemes, and the resulting Green's 
functions are compared to check that they are consistent across normalizations. 

Author: Hugo U. R. Strand (2026)"""


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
        mix=1.,
        normalization='classical',
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

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps, symmetrize=False)
    Delta_w = Gf(mesh=mesh_w, target_shape=[n_orb]*2)

    Delta_w << inverse(iOmega_n)
    #Delta_w << SemiCircular(half_bandwidth=1.0)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    from triqs_xca.block_sparse_solver import BlockSparseSolver

    S = BlockSparseSolver(
        H, beta, w_max, eps, gf_struct, 
        #conserved_operators=[N_tot],
        )

    S.Delta_tau['up'] << Delta_tau
    S.Delta_tau['do'] << Delta_tau

    S.solve(
        max_order=order, tol=ppsc_tol, maxiter=ppsc_maxiter, 
        mix=mix, normalization=normalization)

    S.l = l
    S.order = order
    S.N_tot_exp = S.expectation_value(N_tot)
    print(f'N_tot_exp = {S.N_tot_exp}')

    return S


if __name__ == '__main__':

    opts = dict(
        beta=10.0,
        eps=1e-9,
        ppsc_tol=1e-4,
        ppsc_maxiter=40,
        order=1,
        mix=1.0,
        )

    normalizations = [
        'classic', 
        'ode+classic', 
        'odeG+classic',
        'root',
        'ode+root', 
        'odeG+root',
        ]

    Ss = []
    for normalization in normalizations:
        print(f'Normalization: {normalization}')
        S = solve_slater_condon_bethe_half_filling(l=0, Fs=[3.0], normalization=normalization, **opts)
        print(f'S.diff_G = {S.diff_G}')
        assert(S.diff_G < opts['ppsc_tol'])
        Ss.append(S)

    for S in Ss:
        print(f'eta={S.eta:+1.16E}, iter={S.n_ppsc_iter:3d}, dG={S.diff_G:2.2E} : {S.normalization}')

    for S in Ss[1:]:
        diff_spgf = np.max(np.abs(S.G_tau['up'].data - Ss[0].G_tau['up'].data))
        print(f'diff_spgf = {diff_spgf:2.2E} : {S.normalization} vs. {Ss[0].normalization}')
        assert(diff_spgf < opts['ppsc_tol'])
