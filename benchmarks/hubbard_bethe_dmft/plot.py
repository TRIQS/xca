
import glob
import numpy as np

from h5 import HDFArchive

from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

from triqs_xca.block_sparse_solver import BlockSparseSolver


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])

if __name__ == '__main__':

    filenames = glob.glob('data_hubbard_bethe_dmft_*.h5')
    print(f'filenames = {filenames}')

    rs = []
    for filename in filenames:
        print(f'--> Loading: {filename}')
        with HDFArchive(filename, 'r') as A:
            Sss = A['Sss']
        
        for Ss in Sss:
            r = ListDummy(Ss)
            rs.append(r)


    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 9))
    subp = [3, 1, 1]

    plt.subplot(*subp); subp[-1] += 1
    for r in rs:
        plt.plot(r.U, r.docc_exp, '.-', alpha=0.5, label=f'order = {r.order[0]}, $\\beta$ = {r.beta[0]}')

    #plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.xlabel('U')
    plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')

    plt.subplot(*subp); subp[-1] += 1
    for r in rs:
        plt.plot(r.U, r.dmft_iter, 'o-', alpha=0.5)
    plt.xlabel('U')
    plt.ylabel('DMFT iterations')

    plt.subplot(*subp); subp[-1] += 1
    for r in rs:
        plt.plot(r.U, r.dmft_diff, 'o-', alpha=0.5)
    plt.xlabel('U')
    plt.ylabel('DMFT sc error')
    plt.semilogy([], [])
    plt.grid(True)

    plt.tight_layout()
    #plt.savefig(f'figure_hubbard_bethe_dmft_order{order}_beta{beta}_Umin{Us.min()}_Umax{Us.max()}.pdf')
    plt.show()