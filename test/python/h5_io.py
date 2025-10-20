################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2024 by H. U.R. Strand
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

from triqs_soehyb.solver import Solver

from triqs.operators import c, c_dag

from h5 import HDFArchive


def test_h5():

    beta = 1.0
    lamb = 100.
    eps = 1e-12
    
    H = c_dag(0,0) * c(0,0)
    fundamental_operators = [ c(0,i) for i in range(1) ]

    S = Solver(beta, lamb, eps, H, fundamental_operators)
        
    filename = 'data_h5_io.h5'
    
    with HDFArchive(filename, 'w') as A: A['S'] = S
    with HDFArchive(filename, 'r') as A: S_ref = A['S']
        
    assert( S == S_ref )
        
    
if __name__ == '__main__':

    test_h5()
