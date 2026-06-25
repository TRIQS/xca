
import copy

from triqs.operators import n
from triqs.gfs import make_gf_dlr_imfreq, make_gf_dlr_imtime, inverse, iOmega_n

from triqs_xca.block_sparse_solver import BlockSparseSolver

def test_block_sparse_copy():

    # Minial solver setup
    
    S = BlockSparseSolver(
         H_loc=1.0*n('0', 0), beta=2.3, w_max=1.0, eps=1e-6, gf_struct=[['0', 1]])
    Delta_w = make_gf_dlr_imfreq(S.Delta_tau['0'])
    Delta_w << inverse(iOmega_n)
    S.Delta_tau['0'] << make_gf_dlr_imtime(Delta_w)
    S.solve(max_order=0)

    # Make a deep copy of the solver

    S_copy = copy.deepcopy(S)

    assert(S_copy == S) # Check equality

    # Check that the copy is a different object
    # with different attributes (not the same references)

    # NB: This can not be done with simple types 
    # like int, float, str, but only with objects like G_tau, G, G0, Sigma, etc.

    attrs = ['H_loc', 'Delta_tau', 'G_tau', 'G', 'G0', 'Sigma', 'atom_diag']

    for attr in attrs:
        a = getattr(S, attr)
        a_copy = getattr(S_copy, attr)
        print(f'Checking {attr}: {a is not a_copy}')
        assert(a is not a_copy)


if __name__ == "__main__":
    test_block_sparse_copy()