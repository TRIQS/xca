
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


def load_data(filename, color, markers=False):
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as A:
        d = Dummy()
        for key in A.keys():
            setattr(d, key, A[key])
        d.color = color
        d.markers = markers

    return d


def plot_g(d, tau, g, ls='-', **kwargs):
    if d.markers:
        plt.plot(tau[::5], g[::5], 'o', color=d.color, ms=3, mfc='none', mew=0.75, **kwargs)
    else:
        plt.plot(tau, g, ls, color=d.color, alpha=0.75, **kwargs)
    
if __name__ == '__main__':

    #filename = 'data_cro_fastdiag_nca_beta_5.0_soc_0.2.h5'
    color_1st, color_2nd = '#0072B2', '#D55E00' # Okabe-Ito blue and vermillion

    filename = 'data_cro_fastdiag_1_beta_5.0_soc_0.2.h5'
    nca = load_data(filename, color_1st)
    nca.label = '1st (fastdiag)'
    nca.TrG = np.sum(np.diag(nca.G_faa[-1]))

    #filename = 'data_cro_fastdiag_oca_beta_5.0_soc_0.2.h5'
    filename = 'data_cro_fastdiag_2_beta_5.0_soc_0.2.h5'
    oca = load_data(filename, color_2nd)
    oca.label = '2nd (fastdiag)'
    oca.TrG = np.sum(np.diag(oca.G_faa[-1]))

    filename = 'data_cro_xcabss_1_beta_5.0_soc_0.2.h5'
    nca_xca = load_data(filename, color_1st, markers=True)
    nca_xca.label = '1st (xca)'
    nca_xca.TrG = np.sum(np.diag(nca_xca.G_faa[-1]))

    filename = 'data_cro_xcabss_2_beta_5.0_soc_0.2.h5'
    oca_xca = load_data(filename, color_2nd, markers=True)
    oca_xca.label = '2nd (xca)'
    oca_xca.TrG = np.sum(np.diag(oca_xca.G_faa[-1]))

    print(f'1st order xca vs fastdiag: max|dg| = {np.max(np.abs(nca_xca.g_faa - nca.g_faa)):2.2E}')
    print(f'2nd order xca vs fastdiag: max|dg| = {np.max(np.abs(oca_xca.g_faa - oca.g_faa)):2.2E}')
    
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
    
    # -- Compare fastdiag and xca: hybridization, pseudo-particle Green's function and self-energy

    titles = {'delta_faa': r'\Delta', 'G_faa': r'G', 'Sigma_faa': r'\Sigma'}

    for ref, xca in [(nca, nca_xca), (oca, oca_xca)]:

        plt.figure(figsize=(3.25*3, 3.5*3))
        subp = [6, 2, 1]

        for d in [ref, xca]:
            print(f'{d.label} Tr[G] = {d.TrG}')

        for key, title in titles.items():
            A, B = getattr(ref, key), getattr(xca, key)
            ij = [ (i, j) for i, j in product(range(A.shape[-1]), repeat=2)
                   if max(np.max(np.abs(A[:, i, j])), np.max(np.abs(B[:, i, j]))) > 1e-12 ]
            print(f'order {ref.order}: max|{key} xca - fastdiag| = {np.max(np.abs(B - A)):2.2E}')

            for flt, part in [(np.real, 'Re'), (np.imag, 'Im')]:
                plt.subplot(*subp); subp[-1] += 1
                for d in [ref, xca]:
                    plot_g(d, np.array([]), np.array([]), label=d.label)
                    for i, j in ij:
                        plot_g(d, d.tau_f, -flt(getattr(d, key)[:, i, j]))
                plt.ylabel(rf'$-\mathrm{{{part}}}\, {title}(\tau)$')
                plt.legend(loc='upper right')

                plt.subplot(*subp); subp[-1] += 1
                for i, j in ij:
                    plt.plot(ref.tau_f, -flt(B[:, i, j] - A[:, i, j]), '-', color='k', alpha=0.75)
                plt.title('xca - fastdiag', fontsize=8)

        for ax in plt.gcf().axes[-2:]:
            ax.set_xlabel(r'$\tau$')

        plt.tight_layout()
        plt.savefig(f'figure_cro_compare_order_{ref.order}.pdf')

    #results = [nca, nca_ref, oca, oca_ref]
    results = [nca, oca, nca_xca, oca_xca]
    
    plt.figure(figsize=(3.25, 3.5))
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
        plot_g(d, np.array([]), np.array([]), label=f'{d.label} order')

    for d in results:
        plot_g(d, d.tau_f, -d.g_faa[:, 2, 2].real, '--')

    for d in results:
        g_faa = d.g_faa[:, 0, 0] + d.g_faa[:, 1, 1]
        g_faa *= 0.5
        plot_g(d, d.tau_f, -g_faa.real, '-')
        
    plt.plot([], [], '-', color='gray', label=r'$-( G_{xz} + G_{yz} ) / 2$')
    plt.plot([], [], '--', color='gray', label=r'$-G_{xy}$')
        
    plt.xlabel(r'$\tau$')
    plt.ylim([0, 1.0])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 4, 2, 3, 5] # fastdiag in the first column, xca in the second
    plt.legend([handles[i] for i in order], [labels[i] for i in order],
               fontsize=7, ncol=2, columnspacing=0.8, handlelength=1.5)

    #subp = [2, 2, 3]
    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[1,0])

    for d in results:
        plot_g(d, d.tau_f, -d.g_faa[:, 0, 1].imag, '-', lw=1.0)

    #plt.plot([], [], '--', color='gray', label=r'$- Im[ G_{xz, yz} ]$')
    plt.xlabel(r'$\tau$', labelpad=-7)
    #plt.legend(fontsize=8)
    plt.title(r'- Im[$G_{xz, yz}$]', fontsize=8)
    
    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[1,1])

    for d in results:
        g_faa = d.g_faa[:, 0, 0] - d.g_faa[:, 1, 1]
        g_faa *= 0.5
        plot_g(d, d.tau_f, -g_faa.real, '-', lw=1.0)

        
    #plt.plot([], [], '-', color='gray', label=r'$-( G_{xz} - G_{yz} )$')
    plt.xlabel(r'$\tau$', labelpad=-7)
    #plt.legend(fontsize=8)
    plt.title(r'$-( G_{xz} - G_{yz} )$', fontsize=8)

    #plt.tight_layout()

    plt.savefig('figure_cro_with_soc.pdf')
    
    plt.show()
    
