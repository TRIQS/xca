import numpy as np
from triqs.operators import c, c_dag, n
from triqs.atom_diag import AtomDiag
from h5 import HDFArchive
import atom_diag_utils as utils

if __name__ == "__main__":
    # parameters to tune 
    norb = 2
    all_sym = False  # True for all symmetries, False for just particle number symmetry
    if all_sym:
        h5_fname = "spin_flip_fermion_all_sym.h5"
    else:
        h5_fname = "spin_flip_fermion.h5"

    N = 0
    for kap in range(norb):
        for sig in ['up', 'do']:
            N += n(sig, kap)

    sym_ops = [N]

    # fixed parameters
    beta = 2.0
    mu = 0.25
    U = 1.0
    V = 0.1

    # construct Hamiltonian, fundamental operator set, AtomDiag object
    H = 0
    fops = []
    for i in range(0, norb):
        H += U * n('up', i) * n('do', i) + mu * ( n('up', i) + n('do', i) ) + \
            V * ( c_dag('up', i) * c('do', i) + c_dag('do', i) * c('up', i) )
        fops += [ ('do', i) ]
    for i in range(0, norb):
        fops += [ ('up', i)]
    if all_sym:
        ad = AtomDiag(H, fops)
    else:
        ad = AtomDiag(H, fops, sym_ops)

    # look at c_connection and cdag_connection of ad
    # find groups that are the same
    row_groups = np.zeros(2 * norb, dtype = int) 
    # for the zeroth column, mark the rows that have the same first element
    counter = 1 # increments every time a new group is found
    for s in range(0, ad.n_subspaces):
        oidx0 = 1
        found_oidx0 = False
        for i in range(0, 2 * norb):
            if not found_oidx0 and row_groups[i] != i:
                oidx0 = i
                found_oidx0 = True
        for oidx in range(oidx0, 2 * norb):
            found_pair = False # gets set to True if a pair is found
            for oidx2 in range(0, oidx): # loop over the previous rows
                try:
                    x = ad.c_connection(oidx, s)
                    y = ad.c_connection(oidx2, s)
                except Exception as e:
                    print(f"Failed at oidx={oidx}, s = {s}: {e}")
                if not found_pair and x == y:
                    row_groups[oidx] = row_groups[oidx2] # set the row group to a previous one
                    found_pair = True
            if not found_pair: # no pairs found, so create a new group
                row_groups[oidx] = counter
                counter += 1
    num_sym_sets = len(set(row_groups))
    # get Hamiltonian as a matrix
    H_mat = utils.get_full_h_atomic(ad)
    
    # permute this matrix so that it matches the blocks found by atom_diag
    H_perm = [] # permutation to apply to the rows and columns of H
    H_mat_blocks = [] # array for the blocks of H 
    H_mat_block_inds = np.zeros(ad.n_subspaces, np.int64) # array for block-column indices of H
    for s, state in enumerate(ad.fock_states): # loop over Fock state ordering that induces block structure
        H_perm = np.concatenate((H_perm,state)) 
        H_block = H_mat[state][:,state].reshape(len(state), len(state))
        H_mat_blocks = H_mat_blocks + [H_block]
        if np.all(np.abs(H_block) < 1e-16):
            H_mat_block_inds[s] = -1 # set -1 if block-column has no nonzero block
    H_perm = H_perm.astype(np.int64)
    H_mat_perm = H_mat[H_perm][:,H_perm]

    # need num_sym_sets copies of these
    # c_blocks = [[] for _ in range(ad.n_subspaces)]
    # cdag_blocks = [[] for _ in range(ad.n_subspaces)]
    c_blocks = [[[] for _ in range(ad.n_subspaces)] for _ in range(num_sym_sets)]
    cdag_blocks = [[[] for _ in range(ad.n_subspaces)] for _ in range(num_sym_sets)]

    # orbital indices: do 0, do 1, up 0, up 1
    # save blocks of creation and annihilation operators
    for oidx in range(2 * norb):
        for sidx in range(ad.n_subspaces):
            cidx = ad.c_connection(oidx, sidx)
            fock_idx = np.ix_(ad.fock_states[cidx], ad.fock_states[sidx])
            # c_blocks[sidx] = c_blocks[sidx] + [utils.get_full_c_matrix(ad, oidx)[fock_idx]]
            c_blocks[row_groups[oidx]][sidx] = c_blocks[row_groups[oidx]][sidx] + [utils.get_full_c_matrix(ad, oidx)[fock_idx]]
            didx = ad.cdag_connection(oidx, sidx)
            fock_idx = np.ix_(ad.fock_states[didx], ad.fock_states[sidx])
            # cdag_blocks[sidx] = cdag_blocks[sidx] + [utils.get_full_cdag_matrix(ad, oidx)[fock_idx]]
            cdag_blocks[row_groups[oidx]][sidx] = cdag_blocks[row_groups[oidx]][sidx] + [utils.get_full_cdag_matrix(ad, oidx)[fock_idx]]

    for gidx in range(num_sym_sets):
        for sidx in range(ad.n_subspaces):
            c_blocks[gidx][sidx] = np.array(c_blocks[gidx][sidx])
            cdag_blocks[gidx][sidx] = np.array(cdag_blocks[gidx][sidx])

    # save dense creation and annihilation operators
    c_dense = np.zeros((2 * norb, 4 ** norb, 4 ** norb))
    cdag_dense = np.zeros((2 * norb, 4 ** norb, 4 ** norb))
    for oidx in range(2 * norb):
        c_dense[oidx, :, :] = utils.get_full_c_matrix(ad, oidx)[H_perm][:, H_perm]
        cdag_dense[oidx, :, :] = utils.get_full_cdag_matrix(ad, oidx)[H_perm][:, H_perm]

    print("Number of blocks: ", ad.n_subspaces)
    H_mat_block_sizes = [H_mat_blocks[i].shape[0] for i in range(ad.n_subspaces)]
    import collections
    H_mat_block_sizes = collections.Counter(H_mat_block_sizes)
    print("Block sizes: ", H_mat_block_sizes)
    
    with HDFArchive("/home/paco/feynman/soehyb/test/c++/h5/" + h5_fname) as ar:
        ar['norb'] = norb
        ar['num_blocks'] = ad.n_subspaces
        ar['ad'] = ad
        ar['H_mat_blocks'] = H_mat_blocks
        ar['H_mat_block_inds'] = H_mat_block_inds
        ar['c_blocks'] = c_blocks
        ar['cdag_blocks'] = cdag_blocks
        ar['H_mat_dense'] = H_mat_perm
        ar['c_dense'] = c_dense
        ar['cdag_dense'] = cdag_dense
        ar['num_sym_sets'] = num_sym_sets
        ar['sym_set_labels'] = row_groups
