
# Pseudo particle propagator representation in Python

# In C++ they are instances of the BlockDiagOpFun class

# Q: Should we wrap BlockDiagOpFun in Python or use a list of 3D ndarrays.

G_ppsc = # list of ndarrays ?

Sigma_XCA (list of ndarrays)

ito = itops(eps=1e-12, lambda=100.)

ad = AtomDiag(h_loc)

delta_paa, w_p = AAA_hybridiation_fit(tau_i, delta_iaa)

from block_sparse import SigmaDiagramEvaluator

d = SigmaDiagramEvaluator(w_p, delta_paa, itops, beta, G_ppsc, ad)

topology = [(1, 3), (2, 4)]

Sigma_ppsc = d.compute_diagram(topoly)

# Q: Should we expose the BackBone Class or only deal with topologies on the python side?

N = d.get_number_of_back_bones(topology)

for n in range(N): # MPI parallelize in Python
    Sigma_ppsc += d.compute_diagram(n, topoly)
    
