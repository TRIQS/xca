
import glob
from itertools import product
import numpy as np

from h5 import HDFArchive

from triqs.gf import Gf
from triqs.operators import c, c_dag, n, Operator
from triqs.plot.mpl_interface import oplot, plt, oplotr

import matplotlib.pyplot as plt
    
class Dummy():
    def __init__(self): pass


def load_data(filename):
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as A:
        d = Dummy()
        for key in A.keys():
            setattr(d, key, A[key])
        d.color = plt.plot([], [])[0].get_color()

    return d
    
if __name__ == '__main__':

    plt.figure(figsize=(3.25, 3.5))

    if False:
        plt.figure(figsize=(3.25*3, 3.5*3))
        subp = [6, 2, 1]

    #filename = 'data_cro_fastdiag_nca_beta_5.0_soc_0.2.h5'
    filename = 'data_cro_fastdiag_1_beta_5.0_soc_0.2.h5'
    nca = load_data(filename)
    nca.style = '-'
    nca.label = '1st (fastdiag)'
    nca.TrG = np.sum(np.diag(nca.G_faa[-1]))

    #filename = 'data_cro_fastdiag_oca_beta_5.0_soc_0.2.h5'
    filename = 'data_cro_fastdiag_2_beta_5.0_soc_0.2.h5'
    oca = load_data(filename)
    oca.style = ':'
    oca.label = '2nd (fastdiag)'
    oca.TrG = np.sum(np.diag(oca.G_faa[-1]))
    
    #filename = 'data_cro_ppsc_nca_beta_5.0_soc_0.2.h5'
    #nca_ref = load_data(filename)
    #nca_ref.style = '--'
    #nca_ref.label = '1st (pyppsc)'
    #nca_ref.TrG = np.sum(np.diag(nca_ref.G_faa[-1]))

    #filename = 'data_cro_ppsc_oca_beta_5.0_soc_0.2.h5'
    #oca_ref = load_data(filename)
    #oca_ref.style = ':'
    #oca_ref.label = '2nd (pyppsc)'
    #oca_ref.TrG = np.sum(np.diag(oca_ref.G_faa[-1]))

    #print(f'nca diff {np.max(np.abs(nca.g_faa - nca_ref.g_faa))}')
    #print(f'oca diff {np.max(np.abs(oca.g_faa - oca_ref.g_faa))}')
    
    plt.gca().remove()

    if False:
        results = [oca, oca_ref]
        #results = [oca]
        #results = [nca, nca_ref]

        diff = Dummy()
        diff.G_faa = nca.G_faa - nca_ref.G_faa
        diff.Sigma_faa = nca.Sigma_faa - nca_ref.Sigma_faa
        diff.delta_faa = nca.delta_faa - nca_ref.delta_faa
        diff.tau_f = nca.tau_f
        diff.style = '-'
        diff.color = 'r'

        for flt in [np.real, np.imag]:
            plt.subplot(*subp); subp[-1] += 1
            for d in results:
                plt.plot([], [], d.style, color=d.color, label=d.label)
                for i, j in product(range(d.delta_faa.shape[-1]), repeat=2):
                    plt.plot(d.tau_f, -flt(d.delta_faa[:, i, j]), d.style, color=d.color, alpha=0.75)

            plt.legend(loc='upper right')

            plt.subplot(*subp); subp[-1] += 1
            d = diff
            for i, j in product(range(d.delta_faa.shape[-1]), repeat=2):
                plt.plot(d.tau_f, -flt(d.delta_faa[:, i, j]), d.style, color=d.color, alpha=0.75)


        for flt in [np.real, np.imag]:
            plt.subplot(*subp); subp[-1] += 1
            for d in results:
                print(f'{d.label} Tr[G] = {d.TrG}')
                plt.plot([], [], d.style, color=d.color, label=d.label)
                for i, j in product(range(d.Sigma_faa.shape[-1]), repeat=2):
                    plt.plot(d.tau_f, -flt(d.G_faa[:, i, j]), d.style, color=d.color, alpha=0.75)

            plt.legend(loc='upper right')

            plt.subplot(*subp); subp[-1] += 1
            d = diff
            for i, j in product(range(d.Sigma_faa.shape[-1]), repeat=2):
                plt.plot(d.tau_f, -flt(d.G_faa[:, i, j]), d.style, color=d.color, alpha=0.75)

        for flt in [np.real, np.imag]:
            plt.subplot(*subp); subp[-1] += 1
            for d in results:
                plt.plot([], [], d.style, color=d.color, label=d.label)
                for i, j in product(range(d.Sigma_faa.shape[-1]), repeat=2):
                    plt.plot(d.tau_f, -flt(d.Sigma_faa[:, i, j]), d.style, color=d.color, alpha=0.75)

            plt.subplot(*subp); subp[-1] += 1
            d = diff
            for i, j in product(range(d.Sigma_faa.shape[-1]), repeat=2):
                plt.plot(d.tau_f, -flt(d.Sigma_faa[:, i, j]), d.style, color=d.color, alpha=0.75)

        plt.tight_layout()
        #plt.show()

        #exit()

    #results = [nca, nca_ref, oca, oca_ref]
    results = [nca, oca]
    
    #plt.figure(figsize=(3.25, 3.5))
    subp = [2, 1, 1]

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 1],
        height_ratios=[1, 0.3],
        wspace=0.8, hspace=0.4,
        bottom=0.10, top=0.98,
        left=0.20, right=0.98,
        )    

    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[0,:])
    
    for d in results:
        plt.plot([], [], '-', color=d.color, label=f'{d.label} order')

    for d in results:
        plt.plot(d.tau_f, -d.g_faa[:, 2, 2].real, '--', color=d.color, alpha=0.75)

    for d in results:
        g_faa = d.g_faa[:, 0, 0] + d.g_faa[:, 1, 1]
        g_faa *= 0.5
        plt.plot(d.tau_f, -g_faa.real, '-', color=d.color, alpha=0.75)
        
    plt.plot([], [], '-', color='gray', label=r'$-( G_{xz} + G_{yz} ) / 2$')
    plt.plot([], [], '--', color='gray', label=r'$-G_{xy}$')
        
    plt.xlabel(r'$\tau$')
    plt.ylim([0, 1.0])

    plt.legend(fontsize=8)

    #subp = [2, 2, 3]
    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[1,0])

    for d in results:
        plt.plot(d.tau_f, -d.g_faa[:, 0, 1].imag, '-', color=d.color, lw=1.0, alpha=0.75)

    #plt.plot([], [], '--', color='gray', label=r'$- Im[ G_{xz, yz} ]$')
    plt.xlabel(r'$\tau$', labelpad=-7)
    #plt.legend(fontsize=8)
    plt.title(r'- Im[$G_{xz, yz}$]', fontsize=8)
    
    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[1,1])

    for d in results:
        g_faa = d.g_faa[:, 0, 0] - d.g_faa[:, 1, 1]
        g_faa *= 0.5
        plt.plot(d.tau_f, -g_faa.real, '-', color=d.color, lw=1.0, alpha=0.75)

        
    #plt.plot([], [], '-', color='gray', label=r'$-( G_{xz} - G_{yz} )$')
    plt.xlabel(r'$\tau$', labelpad=-7)
    #plt.legend(fontsize=8)
    plt.title(r'$-( G_{xz} - G_{yz} )$', fontsize=8)

    #plt.tight_layout()

    plt.savefig('figure_cro_with_soc.pdf')
    
    plt.show()
    
