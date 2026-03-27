
import glob
import numpy as np

from h5 import HDFArchive

from triqs.gf import make_gf_imtime
from triqs_xca.block_sparse_solver import BlockSparseSolver

filenames = np.sort(glob.glob('data_*.h5'))

Ss = []
for filename in filenames:
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as ar:
        S = ar['S']
    S.G_tau_fine = make_gf_imtime(S.G_tau, n_tau=400)
    Ss.append(S)

from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

for S in Ss:
    c = plt.plot([], [])[0].get_color()
    oplotr(-S.G_tau['up'][0, 0], label=f'l = {S.l}, order = {S.order}', color=c)
    oplotr(-S.G_tau_fine['up'][0, 0], label=None, color=c)

plt.ylabel(r'$-G(\tau)$')
plt.xlabel(r'$\tau$')
plt.legend(loc='best')
plt.show()