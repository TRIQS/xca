
""" Spin-orbit coupling in t2g cubic harmonic basis following

https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.085150

Author: Hugo U. R. Strand """


import numpy as np

from itertools import product

from triqs.operators import c, c_dag, n, Operator


class t2g_operators:
    
    def __init__(self):

        from triqs_tprf.OperatorUtils import get_quadratic_operator

        norb = 3
        spin_names = ('up','do')
        orb_names = list(range(norb))

        fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
        fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]

        sigma_x = np.array([[0., 1.], [1., 0.]])
        sigma_y = np.array([[0., -1.j], [1.j, 0.]])
        sigma_z = np.array([[1., 0.], [0., -1.]])
        sigma_vec = np.array([sigma_x, sigma_y, sigma_z])

        I_orb = np.eye(norb)
        self.s_x = np.kron(sigma_x, I_orb)
        self.s_y = np.kron(sigma_y, I_orb)
        self.s_z = np.kron(sigma_z, I_orb)
        self.S_x = get_quadratic_operator(self.s_x, fundamental_operators)
        self.S_y = get_quadratic_operator(self.s_y, fundamental_operators)
        self.S_z = get_quadratic_operator(self.s_z, fundamental_operators)
        self.S_vec = [self.S_x, self.S_y, self.S_z]
        self.S2 = sum([ S_i*S_i for S_i in self.S_vec ])

        l_x = np.zeros((3, 3), dtype=complex)
        l_x[0, 2] = -1.j
        l_x[2, 0] = +1.j
        
        l_y = np.zeros((3, 3), dtype=complex)
        l_y[1, 2] = +1.j
        l_y[2, 1] = -1.j
        
        l_z = np.zeros((3, 3), dtype=complex)
        l_z[0, 1] = +1.j
        l_z[1, 0] = -1.j

        l_vec = [l_x, l_y, l_z]

        I_spin = np.eye(2)
        self.l_x = np.kron(I_spin, l_x)
        self.l_y = np.kron(I_spin, l_y)
        self.l_z = np.kron(I_spin, l_z)
        self.L_x = get_quadratic_operator(self.l_x, fundamental_operators)
        self.L_y = get_quadratic_operator(self.l_y, fundamental_operators)
        self.L_z = get_quadratic_operator(self.l_z, fundamental_operators)
        self.L_vec = [self.L_x, self.L_y, self.L_z]
        self.L2 = sum([ L_i*L_i for L_i in self.L_vec ])

        #self.j_x = self.s_x + self.l_x
        #self.j_y = self.s_y + self.l_y 
        #self.j_z = self.s_z + self.l_z
        #self.j_x = np.kron(sigma_x, l_x)
        #self.j_y = np.kron(sigma_y, l_y)
        #self.j_z = np.kron(sigma_z, l_z)
        #self.J_x = get_quadratic_operator(self.j_x, fundamental_operators)
        #self.J_y = get_quadratic_operator(self.j_y, fundamental_operators)
        #self.J_z = get_quadratic_operator(self.j_z, fundamental_operators)
        self.J_x = self.S_x + self.L_x
        self.J_y = self.S_y + self.L_y
        self.J_z = self.S_z + self.L_z
        self.J_vec = [self.J_x, self.J_y, self.J_z]
        self.J2 = sum([ J_i*J_i for J_i in self.J_vec ])

        self.j2 = (0.5**2 + 2**2) * np.eye(6) + \
            2*sum([ np.kron(s, l) for s, l in zip(sigma_vec, l_vec)])
        self.j2_op = get_quadratic_operator(self.j2, fundamental_operators)
        
        
def H_soc_from_levi_cevita(lamb_soc):

    """ Constructs the Triqs operator for spin-orbit coupling
    in the cubic harmonic t2g basis.

    H_{soc} = \sum_{abc} \sum_{s_1, s_2}
        \epsilon_{abc} \sigma^{c}_{s_1, s_2} c^\dagger_{a s_1} c_{b s_2}

    where $a,b,c \in \{x, y, z\}$ and/or $\{yz, xz, xy\}$
    
    The orbital indices [0, 1, 2] are mapped to the cubic harmonics
    according to [0, 1, 2] -> [yz, xz, xy].

    Author: Hugo U. R. Strand """

    norb = 3
    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    #print(f'fundamental_operators = \n{fundamental_operators}')

    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_y = np.array([[0., -1.j], [1.j, 0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    sigma_vec = np.array([sigma_x, sigma_y, sigma_z])
    
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    H_soc = Operator()
    for i, j, k in product(range(norb), repeat=3):
        for s1, s2 in product(range(2), repeat=2):
            S1, S2 = spin_names[s1], spin_names[s2]
            H_soc += lamb_soc * 0.5j * \
                eijk[i, j, k] * sigma_vec[k, s1, s2] * c_dag(S1, i) * c(S2, j)

    return H_soc


if __name__ == '__main__':

    from triqs_tprf.OperatorUtils import quadratic_matrix_from_operator

    lamb_soc = 2.

    norb = 3
    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    fops = [(sn,on) for sn, on in product(spin_names, orb_names)]
    fundamental_operators = [ c(sn,on) for sn,on in product(spin_names, orb_names)]
    print(f'fundamental_operators = \n{fundamental_operators}')

    # -- Order operators according to notation in Eq. (8)

    fundamental_operators = \
        [c('up',1), c('up',0), c('do',2), c('do',1), c('do',0), c('up',2)]

    # This defines the mapping of orbital indices to cubic harmonics
    # according to (see also text before Eq. (8))
    
    # 1 : xz
    # 0 : yz
    # 2 : xy

    # I.e. we have the orbitals in the order: [yz, xz, xy]
    
    H_soc = H_soc_from_levi_cevita(lamb_soc)
    h_soc = quadratic_matrix_from_operator(H_soc, fundamental_operators)
            
    print(f'H_soc =\n{H_soc}')
    print(f'h_soc =\n{h_soc}')

    # Eq. (9),  https://doi.org/10.1103/PhysRevB.97.085150
    
    h1 = np.array([
        [0., -1.j, 1.j],
        [1.j, 0., -1.],
        [-1.j, -1., 0.],
        ])
    
    h2 = np.array([
        [0., 1.j, 1.j],
        [-1.j, 0., 1.],
        [-1.j, 1., 0.],
        ])

    zro = np.zeros((3, 3))
    h_soc_ref = np.bmat([[h1, zro], [zro, h2]])
    print(f'h_soc_ref =\n{h_soc_ref}')

    np.testing.assert_array_almost_equal(h_soc, h_soc_ref)

    # -- Using the expressions for l_i i = {x, y, z}
    # -- in Eqs. 4, 5, 6 in https://doi.org/10.1103/PhysRevB.97.085150
    
    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_y = np.array([[0., -1.j], [1.j, 0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])

    sigma_vec = [sigma_x, sigma_y, sigma_z]

    print(f'sigma_x =\n{sigma_x}')
    print(f'sigma_y =\n{sigma_y}')
    print(f'sigma_z =\n{sigma_z}')

    # -- NB! In the paper l_x and l_y are wrong, they have to be permuted.
    
    l_x = np.zeros((3, 3), dtype=complex)
    l_x[0, 2] = -1.j
    l_x[2, 0] = +1.j
    
    l_y = np.zeros((3, 3), dtype=complex)
    l_y[1, 2] = +1.j
    l_y[2, 1] = -1.j

    l_z = np.zeros((3, 3), dtype=complex)
    l_z[0, 1] = +1.j
    l_z[1, 0] = -1.j

    l_vec = [l_x, l_y, l_z] 

    print(f'l_x =\n{l_x}')
    print(f'l_y =\n{l_y}')
    print(f'l_z =\n{l_z}')
    
    h_soc_ref2 = -sum([ np.kron(s, l) for s, l in zip(sigma_vec, l_vec) ])

    # -- Swap xy_up and xy_down
    h_soc_ref2[[2, 5], :] = h_soc_ref2[[5, 2], :]
    h_soc_ref2[:, [2, 5]] = h_soc_ref2[:, [5, 2]]

    print(f'h_soc_ref2 =\n{h_soc_ref2}')
    np.testing.assert_array_almost_equal(h_soc, h_soc_ref2)
