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

import numpy as np

from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps
from triqs_soehyb.impurity import Fastdiagram

beta = 1.0
lamb = 100.0
eps = 1e-10

n = 4
nops = 1
F = np.zeros((nops, n, n), dtype=complex)
F[:] = np.eye(n)[None, ...]
F_dag = F

dlr_rf = build_dlr_rf(lamb, eps)
ito = ImTimeOps(lamb, dlr_rf)
diagramsolver = Fastdiagram(beta, lamb, ito, F, F_dag)
