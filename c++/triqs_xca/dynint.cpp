

#include "triqs_xca/dynint.hpp"


namespace triqs_xca::dynint {

    using cppdlr::_;

    using nda::range;

    template <bool IsComplex>
    DenseFSet get_operators_and_interactions_dense_impl(
        const triqs_atom_diag_t<IsComplex> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops) {

        assert( hyb_coeffs.extent(0) == dynint_coeffs.extent(0) );
        assert( hyb_coeffs.extent(1) == hyb_coeffs.extent(2) );
        assert( dynint_coeffs.extent(1) == dynint_coeffs.extent(2) );
        
        int r = hyb_coeffs.extent(0);
        int n_hyb = hyb_coeffs.extent(1);
        int n_dynint = dynint_coeffs.extent(1);
        int n_ext = n_hyb + n_dynint;

        auto ext_coeffs = nda::array<dcomplex, 3>::zeros({r, n_ext, n_ext});
        
        ext_coeffs(_, range(0, n_hyb), range(0, n_hyb)) = hyb_coeffs;
        ext_coeffs(_, range(n_hyb, n_ext), range(n_hyb, n_ext)) = dynint_coeffs;

        auto [Fs, Fdags] = atom_diag::get_operators_dense(ad);

        // Create DenseFSet with interactions included as additional operators

        int N = ad.get_full_hilbert_space_dim();

        nda::array<dcomplex, 3> Fs_ext = nda::zeros<dcomplex>(n_ext, N, N);
        nda::array<dcomplex, 3> Fdags_ext = nda::zeros<dcomplex>(n_ext, N, N);

        Fs_ext(_, range(0, n_hyb), range(0, n_hyb)) = Fs;
        Fdags_ext(_, range(0, n_hyb), range(0, n_hyb)) = Fdags;

        // Check atom diag subspaces.
        assert( ad.n_subspaces() == 1 );

        auto U = ad.get_unitary_matrix(0);

        for (int i = 0; i < dynint_ops.size(); ++i) {
            auto op = dynint_ops[i];
            auto op_mat = U * ad.get_op_mat(op).block_mat[0] * nda::conj(nda::transpose(U));
            Fs_ext(n_hyb + i, _, _) = op_mat;
            Fdags_ext(n_hyb + i, _, _) = nda::conj(nda::transpose(op_mat));
        }
        
        return DenseFSet{Fs_ext, Fdags_ext, ext_coeffs};
    }

    DenseFSet get_operators_and_interactions_dense(
        const triqs_atom_diag_t<true> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops) {
        return get_operators_and_interactions_dense_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

    DenseFSet get_operators_and_interactions_dense(
        const triqs_atom_diag_t<false> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops) {
        return get_operators_and_interactions_dense_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

}