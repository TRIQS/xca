
from triqs.gf import inverse, iOmega_n, MeshDLRImFreq, Gf, make_gf_dlr_imtime

from triqs_xca.triqs_solver import TriqsSolver

from triqs_xca.block_sparse_solver import BlockSparseSolver


def test_block_sparse_self_cons():

    beta = 3.0
    mu = 0.0
    e1 = 1.2
    
    eps = 1e-12
    w_max = 10.0 * beta
    tol = 1e-6

    gf_struct = [['0', 1], ['1', 1]]
    fops = [ ('0', 0), ('1', 0) ]

    from triqs.operators import n
    
    N_op = n('0', 0) + n('1', 0)
    
    H = -mu * N_op

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])

    Delta_w << 0.5 * inverse(iOmega_n - e1)
    Delta_tau = make_gf_dlr_imtime(Delta_w)

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)
    S.Delta_tau['0'] << Delta_tau
    S.Delta_tau['1'] << Delta_tau
    S.solve(h_int=H, order=1, tol=tol, compress_hybridization=True)


    BSS = BlockSparseSolver(H, beta, w_max, eps, gf_struct, conserved_operators=[])
    
    BSS.Delta_tau['0'] << Delta_tau
    BSS.Delta_tau['1'] << Delta_tau

    from triqs.gf import make_gf_dlr_imfreq
    Delta_iw = make_gf_dlr_imfreq(BSS.Delta_tau) # BlockGf
    
    BSS.fit_hybridization(tol=tol)

    print(f'BSS.hyb.poles = {BSS.hyb.poles}')
    print(f'BSS.hyb.coefficients =\n{BSS.hyb.coefficients}')

    BSS.init_diagram_evaluator()
    BSS.solve(max_order=1, tol=tol)


if __name__ == '__main__':

    test_block_sparse_self_cons()