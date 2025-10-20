################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2023 by H. U.R. Strand
#
# triqs_soehyb is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

""" Bold hybridization expansion diagram generator

Author: Hugo U. R. Strand (2023) """


from copy import copy

from functools import reduce

import numpy as np


def flatten_pairing_to_list(pairing):
    return reduce(lambda x, y: list(x) + list(y), pairing)


def pop_pair(vec, i, j):
    assert( i != j )
    indices = [i, j]
    p = tuple([ vec[idx] for idx in indices ])
    rest = copy(vec)
    for idx in sorted(indices, reverse=True):
        del rest[idx]
    return p, rest


def _recursive_pairing(parity, partial_pairing, unpaired):

    assert( len(unpaired) % 2 == 0 ) # even number of unpaired indices
    
    if len(unpaired) == 0:
        yield parity, partial_pairing

    for i in range(1, len(unpaired)):
        p, rest = pop_pair(unpaired, 0, i)
        new_partial_pairing = partial_pairing + [p]
        new_parity = -parity if i % 2 == 0 else parity
        yield from _recursive_pairing(new_parity, new_partial_pairing, rest)


def all_pairings(order):    
    unpaired = [ i for i in range(2*order) ]
    yield from _recursive_pairing(1, [], unpaired)


def all_connected_pairings(order, ks=[1]):
    for parity, pairing in all_pairings(order):
        if is_connected(pairing, ks=ks):
            yield parity, pairing


def is_crossing(p1, p2):
    assert( p1[0] < p1[1] )
    assert( p2[0] < p2[1] )
    return \
        (p1[0] < p2[0] and p2[0] < p1[1] and p1[1] < p2[1]) or \
        (p2[0] < p1[0] and p1[0] < p2[1] and p2[1] < p1[1])


def is_valid_ks(ks):
    ks = np.asarray(ks)
    return (ks >= 0).all() and  (np.diff(ks) >= 0).all()


def interval(vertex, ks):
    assert( is_valid_ks(ks) )
    for interval, k in enumerate(ks):
        if vertex <= k - 1:
            return interval
    return interval + 1


def is_ks_connector(pair, ks):
    #k = k - 1
    #return (pair[0] <= k and pair[1] > k) or (pair[1] <= k and pair[0] > k)
    return interval(pair[0], ks=ks) != interval(pair[1], ks=ks)


def split_pairing(pairing, ks):
    rest, connectors = [], []
    for pair in pairing:
        if is_ks_connector(pair, ks=ks):
            connectors.append(pair)
        else:
            rest.append(pair)
    return connectors, rest


def is_connected(pairing, ks=[1]):

    connected = []
    stack, disconnected = split_pairing(pairing, ks)
    
    #pairing = copy(pairing)
    #stack = [pairing.pop(0)]
    #disconnected = pairing    
    
    while len(stack) > 0:
        p_c = stack.pop(0)
        connected.append(p_c)
        for idx, p in reversed(list(enumerate(disconnected))):
            if is_crossing(p_c, p):
                disconnected.pop(idx)
                stack.append(p)

    return len(disconnected) == 0


def sigma_to_gf_diag(pairing):
    """ remove pair that contains the last vertex index """

    n = len(pairing)
    last_vert = 2*n - 1
    idx = next( x for x in range(n) if last_vert in pairing[x] )
    p = list(pairing.pop(idx))
    idx = p.index(last_vert)
    p.pop(idx)
    k = p[0]

    shift = lambda idx: idx if idx < k else idx - 1            
    pairing = [(shift(x), shift(y)) for x, y in pairing]            
    return k, pairing


def all_gf_pairings(order):

    parity_pairs = [ x for x in all_connected_pairings(order+1) ]
    print(f'n_diags = {len(parity_pairs)}')

    from collections import defaultdict
    diags = defaultdict(list)
    for parity, pairing in parity_pairs:
        k, pairing = sigma_to_gf_diag(pairing)
        diags[k].append((parity, pairing))
    
    return diags


if __name__ == '__main__':

    exit()
    
    #for order in [1]:
    for order in [1, 2, 3, 4]:
        print('-'*72)
        print(f'order = {order}')
        print('-'*72)
        for par, pair in all_connected_pairings(order):
            print(f'{par:+d}, {pair}')

    #exit()
                    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    
    def plot_pairs(ax, pairs, ks=[]):

        colors = dict()
        
        #for k in ks:
        for i1 in range(len(ks)+1):
            for i2 in range(len(ks)+1):
                if i1 != i2:
                    c = ax.plot([], [])[0].get_color()
                else:
                    c = 'gray'
                colors[(i1, i2)] = c
        
        # arcs
        for p in pairs:
            r = p[1] - p[0]
            c = 0.5*np.sum(p)
            #color = 'gray'
            #for k in ks:
            #    if is_ks_connector(p, ks=[k]):
            #        color = colors[k]
            color = colors[(interval(p[0], ks), interval(p[1], ks))]
                    
            a = Arc((c, 0), r, r, theta1=0, theta2=180, fill=False, color=color)
            ax.add_patch(a)

        # backbone
        n = np.max([2*len(pairs) - 1, np.max(ks)])
        s = np.min([0, np.min(ks) - 1])
        
        ax.plot([s, n], [0, 0], 'k', lw=2)

        # dashed vertical lines
        x = -1.
        y = 0.
        for k in ks:
            
            x_new = k - 0.5
            if x_new == x:
                y -= 0.25
            else:
                y = 0.
            x = x_new
                
            ax.plot([x], [y], '.r')

        ax.tick_params(bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.spines[:].set_visible(False)
        ax.set_ylim([-0.5*(1+len(ks)), (n+1)//2 - 0.25])


    if False:
        order = 3
        nks = { 1:1, 2:2, 3:7, 4:42 }

        nk = nks[order]
        nr = 2*order - 1
        subp = [nr, nk, 0]

        fig = plt.figure(figsize=(26 * nk/42, 4 * nr/7))

        for k in range(1, 2*order):

            parity_pairs = [ x for x in all_connected_pairings(order, ks=[k]) ]
            n = len(parity_pairs)
            print(f'n = {n}')

            for par, pairs in parity_pairs:
                subp[-1] += 1
                ax = fig.add_subplot(*subp, aspect='equal')
                plot_pairs(ax, pairs, ks=[k])

            subp[-1] = nk * k

        plt.tight_layout()
        plt.savefig(f'figure_order_{order}.pdf')
        plt.show()
        exit()

    if True:
        order = 3
        
        nks = { 1:1, 2:2, 3:7, 4:42 }

        nk = nks[order]
        nr = 2*order - 1
        subp = [nr, nk, 0]

        fig = plt.figure(figsize=(26 * nk/42, 4 * nr/7))

        diags = all_gf_pairings(order)
        
        for k in range(1, 2*order):

            parity_pairs = diags[k]
            n = len(parity_pairs)
            print(f'n = {n}')

            for par, pairs in parity_pairs:
                subp[-1] += 1
                ax = fig.add_subplot(*subp, aspect='equal')
                plot_pairs(ax, pairs, ks=[k])

            subp[-1] = nk * k

        plt.tight_layout()
        plt.savefig(f'figure_order_{order}.pdf')
        plt.show()
        exit()
        
    if False:
        subp = [4, 5, 0]
        fig = plt.figure(figsize=(10, 10))

        order = 2
        for k1 in range(0, 2*order):
            for k2 in range(k1, 2*order+1):
                print(f'k1, k2 = {k1}, {k2}')

                parity_pairs = [ x for x in all_connected_pairings(order, ks=[k1, k2]) ]

                for par, pairs in parity_pairs:
                    subp[-1] += 1
                    ax = fig.add_subplot(*subp, aspect='equal')
                    plot_pairs(ax, pairs, ks=[k1, k2])

                #for parity, pairs in all_connected_pairings(order, ks=[k1, k2]):
                #    print(f'{parity:+d}, {pairs}')

        plt.tight_layout()
        plt.show()
        exit()

    if False:
        subp = [10, 10, 0]
        fig = plt.figure(figsize=(10, 10))

        order = 2
        for k1 in range(0, 2*order):
            for k2 in range(k1, 2*order+1):
                for k3 in range(k2, 2*order+1):
                    print(f'k1, k2, k3 = {k1}, {k2}, {k3}')
                    ks = [k1, k2, k3]
                    parity_pairs = [ x for x in all_connected_pairings(order, ks=ks) ]

                    for par, pairs in parity_pairs:
                        subp[-1] += 1
                        ax = fig.add_subplot(*subp, aspect='equal')
                        plot_pairs(ax, pairs, ks=ks)

        plt.tight_layout()
        plt.show()
        exit()
