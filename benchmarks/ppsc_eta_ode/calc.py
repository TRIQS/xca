import numpy as np

from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime

from triqs_xca.triqs_solver import TriqsSolver
from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse_solver import pseudo_particle_block_gf_to_dense


def test_block_sparse_self_cons(verbose=False):

    beta = 1000.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-10
    w_max = 10.0
    tol = 1e-6

    order = 4
    maxiter = 100

    gf_struct = [['0', 1]]

    from triqs.operators import n
    H = -mu * n('0', 0)

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

    Delta_w << inverse(iOmega_n - e1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    tau_mesh = Delta_tau.mesh
    
    S = BlockSparseSolver(H, beta, w_max, eps, gf_struct)

    S.Delta_tau['0'] << Delta_tau
 
    #S.solve(max_order=order, tol=tol, maxiter=maxiter)

    max_order = 1
    S.fit_hybridization(tol=1e-9)
    #Z = S.partition_function_from_ppgf(S.G)

    S.Sigma = S.eval_pseudo_particle_self_energy(S.G, max_order)
    eta = 0

    from triqs_xca.module import trace, convolve_ppsc

    
    #from scipy.interpolate import InterpolatedUnivariateSpline
    #Z = InterpolatedUnivariateSpline(etas, Zs)
    #print(dir(Z))

    # Testing derivative "flow" for 
    # pseudo-particle chemical potential
    # solving an ode
    # deta / dalpha = f(eta, alpha, Sigma)
    # where (1 G0*(alpha * Sigma - eta))G_alpha = G0
    # starting from alpha = 0, eta = 0, Tr[G0] = -1
    # to alpha = 1 and eta for Sigma!

    sol = S.solve_ppsc_chempot_adiabatic_ode(tol=10*eps)
    #sol = S.solve_ppsc_chempot_adiabatic_ode_3rd(tol=10*eps)
    S.solve_ppsc_chempot_newton(tol=10*eps)
    
    sol.Za = np.array([ S.Z_alpha(alpha, eta) \
        for alpha, eta in zip(sol.t, sol.y[0])])

    alphas = np.linspace(0, 1, num=400)
    sol.Za_vec = np.array([ S.Z_alpha(t, sol.sol(t)[0]) for t in alphas ])

    # Compute Z and dZ/deta vs eta
    etas = np.linspace(np.max(sol.y[0]) * 0.995, np.max(sol.y[0]) * 1.005, num=100)
    Zs = np.array([ S.Z_alpha(1., eta) for eta in etas ]).T
    dZ_detas = np.array([ S.dZ_alpha_deta(1., eta) for eta in etas ]).T

    afix = 0.01
    deta_dalpha = np.array([ S.deta_dalpha(afix, eta) for eta in etas ])
    d2eta_dalpha_deta = np.array([ S.d2eta_dalpha_deta(afix, eta) for eta in etas ])

    etas_f = np.linspace(np.max(sol.y[0]) * 0.995, np.max(sol.y[0]) * 1.005, num=400)

    from scipy.interpolate import InterpolatedUnivariateSpline
    spl_deta_dalpha = InterpolatedUnivariateSpline(etas, deta_dalpha)
    spl_d2eta_dalpha_deta = spl_deta_dalpha.derivative()

    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

    plt.figure(figsize=(8, 8))

    subp = [4, 2, 1]

    plt.subplot(*subp); subp[-1] += 1
    #plt.plot(alphas, Za_vec, '-')
    plt.plot(alphas, sol.Za_vec, '-')
    plt.plot(sol.t, sol.Za, 'o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$Z_\alpha$')
    plt.ylim([-0.5, 1.5])
    plt.grid(True)

    plt.subplot(*subp); subp[-1] += 1
    #plt.plot(alphas, np.abs(Za_vec-1), '-')
    plt.plot(alphas, np.abs(sol.Za_vec-1), '-')
    plt.plot(sol.t, np.abs(sol.Za-1), 'o')
    plt.semilogy([], [])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$|Z_\alpha - 1|$')

    plt.subplot(*subp); subp[-1] += 1
    #plt.plot(alphas, eta_vec, '-')
    plt.plot(alphas, sol.sol(alphas)[0], '-')
    plt.plot(sol.t, sol.y[0], 'o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\eta$')

    plt.subplot(*subp); subp[-1] += 1
    #plt.plot(alphas, eta_vec / alphas, '-')
    plt.plot(alphas, sol.sol(alphas)[0] / alphas, '-')
    plt.plot(sol.t, sol.y[0] / sol.t, 'o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\eta/ \alpha$')

    #plt.subplot(*subp); subp[-1] += 1
    
    plt.subplot(*subp); subp[-1] += 1
    plt.plot(etas, 1 + 0*etas, '-r', lw=4)
    plt.plot(etas, Zs, '.-')
    plt.plot(sol.y[0, -1], 1, 'bx')
    plt.semilogy([], [])
    plt.grid(True)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$Z_\eta$')

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(etas, dZ_detas, '.-', label='dZdeta')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$dZ/d\eta$')
    plt.legend()

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(etas, deta_dalpha, '.')
    plt.plot(etas_f, spl_deta_dalpha(etas_f), '-')
    plt.ylabel(r'$d\eta / d\alpha$')
    plt.xlabel(r'$\eta$')

    plt.subplot(*subp); subp[-1] += 1
    plt.plot(etas, d2eta_dalpha_deta, '.')
    #plt.plot(etas, d2eta_dalpha_deta/2, 'x')
    #plt.plot(etas, d2eta_dalpha_deta - d2eta_dalpha_deta[0], 'x')
    plt.plot(etas_f, spl_d2eta_dalpha_deta(etas_f), '-')
    plt.ylabel(r'$d/d\eta ( d\eta / d\alpha )$')
    plt.xlabel(r'$\eta$')

    plt.tight_layout()
    plt.show()
    exit()

    G_BSS_tau = make_gf_imtime(G_BSS, n_tau=400)
    Sigma_BSS_tau = make_gf_imtime(Sigma_BSS, n_tau=400)
    g_BSS_tau = make_gf_imtime(g_BSS, n_tau=400)

    if verbose:
        plt.figure(figsize=(8, 9))

        subp = [2, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_BSS_tau, '-')  
        oplotr(G_S)
        plt.ylabel(r'$G(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(Sigma_BSS_tau, '-')
        oplotr(Sigma_S)
        plt.ylabel(r'$\Sigma(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(g_BSS_tau, '-')
        oplotr(g_S)
        plt.ylabel(r'$g(\tau)$')

        plt.tight_layout()
        plt.show()

    np.testing.assert_array_almost_equal(G_BSS.data, G_S.data)
    np.testing.assert_array_almost_equal(Sigma_BSS.data, Sigma_S.data)
    np.testing.assert_array_almost_equal(g_BSS['0'].data, g_S['0'].data)


if __name__ == '__main__':

    test_block_sparse_self_cons(verbose=False)