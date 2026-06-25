################################################################################
#
# triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2024 by H. U.R. Strand
#
# triqs_xca is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_xca. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from triqs_xca.block_sparse_solver import BlockSparseSolver

from triqs.operators import n
from triqs.gfs import inverse, iOmega_n, make_gf_dlr_imfreq, make_gf_dlr_imtime

from h5 import HDFArchive


def test_h5():

    beta = 1.0
    w_max = 100.
    eps = 1e-12
    
    H = n('0',0)
    gf_struct = [['0', 1]]

    S = BlockSparseSolver(H, beta, w_max, eps, gf_struct)
    
    Delta_w = make_gf_dlr_imfreq(S.Delta_tau['0'])
    Delta_w << inverse(iOmega_n)
    S.Delta_tau['0'] << make_gf_dlr_imtime(Delta_w)

    S.solve(max_order=1, tol=1e-9, maxiter=1)
        
    filename = 'data_h5_io_bs.h5'
    
    with HDFArchive(filename, 'w') as A: A['S'] = S
    with HDFArchive(filename, 'r') as A: S_ref = A['S']

    assert( S == S_ref )
        
    
if __name__ == '__main__':

    test_h5()
