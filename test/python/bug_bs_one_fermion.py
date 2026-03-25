""" Trigger segfault bug in DiagramEvaluator

Suspected origin is the get_operators(...) functon in atom_diag_utils.cpp

It assumes that the hybridization expansion matrix sizes are __even__
`  int norb = hyb_coeffs.extent(1) / 2; `

Which is not the case for problems with an odd number of fermions, e.g. a single fermion.

Here we test this case and trigger a segfault.

"""

import numpy as np

import triqs.utility.mpi as mpi

beta = 2.0
mu = 1.0

eps = 1e-12
lamb = 20.0 * beta
w_max = lamb / beta

from triqs.gf import Gf, MeshDLRImTime, BlockGf

mesh_tau = MeshDLRImTime(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)

from triqs.operators import n
    
N_op = n('0', 0) 
    
H = -mu * N_op

print(f'H = {H}')

from triqs.atom_diag import AtomDiag

fops = [ ('0', 0) ]

ad = AtomDiag(H, fops, [N_op])
print(ad)

from triqs_xca.block_sparse_solver import atomic_pseudo_particle_greens_function

G_ppsc = atomic_pseudo_particle_greens_function(ad, beta, mesh_tau)
print(G_ppsc)

poles = np.array([0.0, 1.0])
coefficients = np.array([ [[1.0]],  [[0.1]] ], dtype=complex)

print(poles.shape)
print(coefficients.shape)

from triqs_xca.block_sparse import DiagramEvaluator

d = DiagramEvaluator(poles, coefficients, mesh_tau, ad)
