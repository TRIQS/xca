

import itertools
import numpy as np

from triqs_xca.block_sparse import convolve_ppsc as conv

from triqs.gf import MeshDLRImTime, BlockGf
from triqs.gf import make_gf_imtime


def test_const_GG_volterra_integral_DLR(
        beta=1.0, eps=1e-12, w_max=1., verbose=False):

    m = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    tau = np.array([ float(t) for t in m ])
    #print(m)

    G = BlockGf(mesh=m, gf_struct=[['0', 1]])
    #print(G)

    G['0'].data[:] = -1.0

    GG = conv(G, G)

    GG_err = GG.copy()
    GG_err['0'].data[:, 0, 0] -= tau

    err = np.max(np.abs(GG_err['0'].data))
    print(f'beta = {beta:2.2E}, eps = {eps:2.2E}, w_max = {w_max:2.2E}, err = {err:2.2E}, err/eps = {err/eps:2.2E}')

    # NB! We get two or more decimals of accuracy than eps
    # (that is a good -- but surprising -- thing)
    assert( err < 1e-2 * eps ) 
    
    if verbose:
        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

        subp = [3, 1, 1]
        plt.figure(figsize=(5, 7))

        plt.subplot(*subp); subp[-1] += 1
        oplot(G)
        oplot(make_gf_imtime(G, n_tau=400))

        plt.subplot(*subp); subp[-1] += 1
        oplot(GG)
        oplot(make_gf_imtime(GG, n_tau=400))

        plt.subplot(*subp); subp[-1] += 1
        GG_err_fine = make_gf_imtime(GG_err, n_tau=400)

        GG_err['0'].data[:] = np.abs(GG_err['0'].data[:])
        GG_err_fine['0'].data[:] = np.abs(GG_err_fine['0'].data[:])

        oplot(GG_err)
        oplot(GG_err_fine)
        plt.semilogy([], [])

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    w_maxs = [1., 10., 100., 1000., 10000.]
    epss = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

    for w_max, eps in itertools.product(w_maxs, epss):
        test_const_GG_volterra_integral_DLR(w_max=w_max, eps=eps)