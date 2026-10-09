
from itertools import combinations, product

import numpy as np

from h5 import HDFArchive

from triqs.gfs import SemiCircular, make_gf_dlr, make_gf_dlr_imtime, make_gf_dlr_imfreq
from triqs.operators import c, c_dag, n, Operator
from triqs.operators.util.U_matrix import U_matrix_kanamori
from triqs.operators.util.hamiltonians import h_int_kanamori

from pyed.SparseMatrixFockStates import SparseMatrixRepresentation

from t2g_soc import H_soc_from_levi_cevita

from triqs_xca.block_sparse_solver import BlockSparseSolver, is_root


def run_calc(beta, lamb, eps, order,
             mu=3.9530058540332917,
             lamb_soc=0.,
             mix=1.0,
             write_h5=False, plot_flag=True, S_old=None):

    U = 4.6
    J = 0.8
    norb = 3

    half_filling_shift = 0.5*(5*U - 10*J)
    mu += half_filling_shift
    t_diag = np.array([0.5, 0.5, 1.0])
    T_diag = np.concatenate([t_diag, t_diag])
    delta_cf = -1.0

    w_max = lamb / beta

    ppsc_tol = 1e-6
    ppsc_maxiter = 1
    dmft_tol = 1e-6
    dmft_maxiter = 100 # self-cons settings

    spin_names = ('up','do')
    orb_names = list(range(norb))

    KanMat1, KanMat2 = U_matrix_kanamori(norb, U, J)
    H = h_int_kanamori(spin_names, norb, KanMat1, KanMat2, J, off_diag=True)

    N_2 = n('up', 2) + n('do', 2)
    N_up = sum([ n('up', idx) for idx in range(norb) ])
    N_do = sum([ n('do', idx) for idx in range(norb) ])

    H -= mu *(N_up + N_do)
    H += delta_cf * N_2 # using that orbital 2 is xy

    H += H_soc_from_levi_cevita(lamb_soc)

    # SOC mixes spins, map (spin, orb) to a single block with index spin*norb + orb

    block = '0'
    gf_struct = [(block, 2*norb)]
    idx_map = { (sn, on) : (block, sidx*norb + on)
                for (sidx, sn), on in product(enumerate(spin_names), orb_names) }

    H_block = Operator()
    for term, coef in H:
        op = coef
        for dag, idx in term:
            op = op * (c_dag if dag else c)(*idx_map[tuple(idx)])
        H_block += op
    H = H_block

    N_tot = sum([ n(block, idx) for idx in range(2*norb) ])

    # Setup impurity problem

    S = BlockSparseSolver(H, beta, w_max, eps, gf_struct)

    if S_old is None:
        Delta_w = make_gf_dlr_imfreq(S.Delta_tau[block])
        for idx, t_i in enumerate(T_diag):
            Delta_w[idx, idx] << t_i**2 * SemiCircular(half_bandwidth=t_i)
        S.Delta_tau[block] << make_gf_dlr_imtime(Delta_w)
    else:
        S.Delta_tau << S_old.Delta_tau
        for bidx, g in S_old.G:
            S.G[bidx] << g
        S.eta = S_old.eta0 + S_old.eta - S.eta0 # eta is relative to eta0

    for dmft_iter in range(dmft_maxiter):

        S.solve(order, tol=ppsc_tol, maxiter=ppsc_maxiter, mix=mix, hyb_tol=10*eps)

        delta_old = S.Delta_tau[block].copy()
        S.Delta_tau[block].data[:] = np.einsum(
            'a,iab,b->iab', T_diag, S.G_tau[block].data, T_diag)

        delta_diff = np.max(np.abs(S.Delta_tau[block].data - delta_old.data))

        N = S.expectation_value(N_tot).real

        if is_root():
            print(
                f'DMFT: iter {dmft_iter:3d}, ddelta {delta_diff:2.2E}' + \
                f' - N {N:8.8E}')

        if delta_diff < dmft_tol: break

    S.N = N

    tau_i = np.array([ float(t) for t in S.mesh_tau ])
    g_iaa = S.G_tau[block].data
    delta_iaa = S.Delta_tau[block].data

    g_dlr = make_gf_dlr(S.G_tau[block])
    delta_dlr = make_gf_dlr(S.Delta_tau[block])
    rho_aa = -g_dlr(beta)

    tau_f = np.linspace(0, beta, num=100)
    g_faa = np.array([ g_dlr(t) for t in tau_f ])
    delta_faa = np.array([ delta_dlr(t) for t in tau_f ])

    # Pseudo-particle G and Sigma in the pyed Fock basis of the fastdiag script.
    # The map W from the pyed to the triqs Fock basis is built by creating
    # each Fock state from the vacuum with the same c^dag chain in both bases.

    ad = S.atom_diag
    fops = list(ad.fops)
    nf, dim = len(fops), ad.full_hilbert_space_dim

    Cd_t = np.zeros((nf, dim, dim), dtype=complex)
    for k, b in product(range(nf), range(ad.n_subspaces)):
        t = ad.cdag_connection(k, b)
        if t < 0: continue
        Ut, Ub = ad.unitary_matrices[t], ad.unitary_matrices[b]
        Cd_t[k][np.ix_(ad.fock_states[t], ad.fock_states[b])] = \
            Ut @ ad.cdag_matrix(k, b) @ Ub.conj().T

    rep = SparseMatrixRepresentation([ c(*f) for f in fops ])
    Cd_p = np.array([ op.todense() for op in rep.sparse_operators.c_dag ])

    vac_t, vac_p = np.zeros(dim, dtype=complex), np.zeros(dim, dtype=complex)
    vac_t[np.flatnonzero(np.isclose(np.einsum('kab,kab->a', Cd_t.conj(), Cd_t), 0))] = 1
    vac_p[np.flatnonzero(np.isclose(np.einsum('kab,kab->a', Cd_p.conj(), Cd_p), 0))] = 1

    W = np.zeros((dim, dim), dtype=complex)
    for occ in (occ for m in range(nf + 1) for occ in combinations(range(nf), m)):
        st, sp = vac_t, vac_p
        for k in occ: st, sp = Cd_t[k] @ st, Cd_p[k] @ sp
        W += np.outer(st, sp.conj())

    G_faa = np.zeros((len(tau_f), dim, dim), dtype=complex)
    Sigma_faa = np.zeros_like(G_faa)
    for sidx in range(ad.n_subspaces):
        fidx = np.ix_(range(len(tau_f)), ad.fock_states[sidx], ad.fock_states[sidx])
        G_dlr, Sigma_dlr = make_gf_dlr(S.G[sidx]), make_gf_dlr(S.Sigma[sidx])
        G_faa[fidx] = np.array([ G_dlr(t) for t in tau_f ])
        Sigma_faa[fidx] = np.array([ Sigma_dlr(t) for t in tau_f ])
    G_faa = W.conj().T @ G_faa @ W
    Sigma_faa = W.conj().T @ Sigma_faa @ W

    if is_root() and write_h5:
        filename = f'data_cro_xcabss_{order}_beta_{beta}_soc_{lamb_soc}.h5'
        print(f'--> Writing: {filename}')
        with HDFArchive(filename, 'w') as A:

            A['S'] = S

            A['g_iaa'] = g_iaa
            A['delta_iaa'] = delta_iaa
            A['tau_i'] = tau_i

            A['g_faa'] = g_faa
            A['delta_faa'] = delta_faa
            A['tau_f'] = tau_f

            A['G_faa'] = G_faa
            A['Sigma_faa'] = Sigma_faa

            A['rho'] = rho_aa

            A['beta'] = beta
            A['order'] = order

            A['lamb'] = lamb
            A['eps'] = eps

            A['H'] = H

            A['U'] = U
            A['J'] = J
            A['mu'] = mu
            A['t_diag'] = t_diag
            A['delta_cf'] = delta_cf
            A['lamb_soc'] = lamb_soc
            A['N'] = N

    if is_root() and plot_flag:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        subp = [6, 6, 1]

        rng = [0, 1, 5, 3, 4, 2]
        lables = [
            r'$yz \uparrow$',
            r'$xz \uparrow$',
            r'$xy \uparrow$',
            r'$yz \downarrow$',
            r'$xz \downarrow$',
            r'$xy \downarrow$',
            ]
        colors = {np.real: '#0072B2', np.imag: '#E69F00'}
        for a, b in product(rng, repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            la, lb = lables[a], lables[b]
            plt.title(la + ', ' + lb, fontsize=7)
            for flt, color in colors.items():
                plt.plot(
                    tau_i, flt(g_iaa[:, a, b]),
                    '.', color=color, label=f'g {a},{b}', alpha=0.5)
                plt.plot(
                    tau_f, flt(g_faa[:, a, b]),
                    '-', color=color, alpha=0.75)
            if a != b: plt.ylim([-0.1, 0.1])
            else: plt.ylim([-1, 0.05])

        plt.tight_layout()
        plt.savefig(
            f'figure_xcabss_cro_order_{order}_lamb_soc_{lamb_soc}.pdf')
        plt.show()

    return S


def fix_N_calc(N_target, mu0, mu1, func):

    def target_function(mu):
        N = func(mu)
        print('='*72)
        print(f'mu = {mu}, N = {N}')
        print('='*72)
        return N_target - N

    from scipy.optimize import root_scalar

    sol = root_scalar(
        target_function, bracket=[mu0, mu1],
        x0=mu0, x1=mu1, method='bisect', xtol=1e-3)

    return sol.root


if __name__ == '__main__':

    order = 2

    parms = [
        #( 5., 200., 1e-9),
        ( 5., 100., 1e-8),
        #( 5., 1000., 1e-12),
        ]

    N_target = 4.0

    mu_nca = 3.977224604033292
    mu_oca = 3.965505854033292
    mu_tca = 3.9655058540332924

    #mu = mu_nca
    #mu = mu_oca
    mu = mu_tca # NB! Use mu_tca for all orders to compare with prev code.

    mu0 = mu - 0.05
    mu1 = mu + 0.05

    lamb_socs = [0.2]

    for beta, lamb, eps in parms:

        for lamb_soc in lamb_socs:

            if is_root():
                print('='*72)
                print('='*72)
                print(f'lamb_soc = {lamb_soc}')
                print('='*72)
                print('='*72)

            def N_func(mu, write_h5=False):
                S = run_calc(beta, lamb, eps, order,
                             lamb_soc=lamb_soc,
                             mu=mu,
                             write_h5=write_h5)
                return S.N

            #mu = fix_N_calc(N_target, mu0, mu1, N_func)

            if is_root():
                print('='*72)
                print('='*72)
                print('--> Final')
                print(f'mu = {mu}')
                print('='*72)
                print('='*72)

            opts = dict(lamb_soc=lamb_soc, mu=mu, write_h5=True, plot_flag=False)

            S = run_calc(beta, lamb, eps, 1, **opts)
            S = run_calc(beta, lamb, eps, 2, S_old=S, **opts)
            #S = run_calc(beta, lamb, eps, 3, S_old=S, **opts)
