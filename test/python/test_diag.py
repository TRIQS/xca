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

""" Diagram generator tests """


from triqs_soehyb.diag import is_crossing, is_connected
from triqs_soehyb.diag import pop_pair, flatten_pairing_to_list
from triqs_soehyb.diag import all_pairings, all_connected_pairings, all_gf_pairings


# Number of irreducible chord diagrams with 2n nodes. [https://oeis.org/A000699]
n_strongly_irred_pairings = [ 1, 1, 1, 4, 27, 248, 2830, 38232, 593859, 10401712, 202601898, 4342263000 ]


def test_pop_pair():

    vec = [1, 2, 3, 4]
    p, rest = pop_pair(vec, 1, 3)
    assert( p == (2, 4) )
    assert( rest == [1, 3] )


def test_is_crossing():
    assert( is_crossing((0, 1), (2, 3)) == False )
    assert( is_crossing((0, 2), (1, 3)) == True )
    

def test_is_connected():

    assert( is_connected([(0, 1)]) == True )
    assert( is_connected([(0, 1), (2, 3)]) == False )
    assert( is_connected([(0, 2), (1, 3)]) == True )

    assert( is_connected([(0, 2), (1, 4), (3, 5)]) == True )
    assert( is_connected([(0, 3), (1, 4), (2, 5)]) == True )
    assert( is_connected([(0, 3), (1, 5), (2, 4)]) == True )
    assert( is_connected([(0, 4), (1, 3), (2, 5)]) == True )


def test_all_pairings():

    from sympy.combinatorics import Permutation
    from sympy.functions.combinatorial.factorials import factorial2

    print('test_all_pairings')
    
    for order in range(1, 7):
        n_pairings = 0
        for parity, pairing in all_pairings(order):
            perm = Permutation(flatten_pairing_to_list(pairing))
            assert( parity == (-1)**perm.parity() )
            n_pairings += 1

        print(f'order = {order}, n_pairings = {n_pairings}')
        assert( n_pairings == factorial2(2*order-1) )


def test_all_connected_pairings():

    print('test_all_connected_pairings')

    assert( list(all_connected_pairings(1)) == [(+1, [(0, 1)])] )
    assert( list(all_connected_pairings(2)) == [(-1, [(0, 2), (1, 3)])] )
    assert( list(all_connected_pairings(3)) == [        
        (+1, [(0, 2), (1, 4), (3, 5)]),
        (-1, [(0, 3), (1, 4), (2, 5)]),
        (+1, [(0, 3), (1, 5), (2, 4)]),
        (+1, [(0, 4), (1, 3), (2, 5)]),
        ] )
    
    for order in range(1, 7):
        n_pairings = 0
        for parity, pairing in all_connected_pairings(order):
            n_pairings += 1
        print(f'order = {order}, n_pairings = {n_pairings}')
        assert( n_pairings == n_strongly_irred_pairings[order] )


def test_gf_diagrams():

    print('test_gf_diagrams')
    for order in [1, 2, 3, 4, 5]:
        print(f'order = {order}')

        diags = all_gf_pairings(order)

        assert( sorted(list(diags.keys())) == list(range(1, 2*order)) )
        
        for k in range(1, 2*order):

            parity_pairs_ref = diags[k]
            parity_pairs = [ x for x in all_connected_pairings(order, ks=[k]) ]

            pairs_ref = set([ tuple(pp[1]) for pp in parity_pairs_ref ])
            pairs = set([ tuple(pp[1]) for pp in parity_pairs ])

            #print(f'order = {order}, k = {k}')
            #print(f'{pairs}')
            #print(f'{pairs_ref}')

            assert( pairs == pairs_ref )
        

if __name__ == '__main__':
    test_pop_pair()
    test_is_crossing()
    test_is_connected()
    test_all_pairings()
    test_all_connected_pairings()
    test_gf_diagrams()
