#include <stdexcept>
#include <string>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/dense/dynint.hpp"

namespace triqs_xca::dense::dynint {

    using cppdlr::_;

    using nda::range;

    template <bool IsComplex>
    FSet get_operators_and_interactions_impl(
        const triqs_atom_diag_t<IsComplex> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {

        auto ext_coeffs = hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs);
        int n_ext = ext_coeffs.extent(1);

        auto [Fs, Fdags] = atom_diag::get_operators(ad);

        // Create FSet with interactions included as additional operators

        int N = ad.get_full_hilbert_space_dim();

        nda::array<dcomplex, 3> Fs_ext = nda::zeros<dcomplex>(n_ext, N, N);
        nda::array<dcomplex, 3> Fdags_ext = nda::zeros<dcomplex>(n_ext, N, N);

        int n_hyb = hyb_coeffs.extent(1);
        
        Fs_ext(range(0, n_hyb), _, _) = Fs;
        Fdags_ext(range(0, n_hyb), _, _) = Fdags;

        // Check atom diag subspaces, the interaction operators are read from subspace 0 only
        if (ad.n_subspaces() != 1)
            throw std::invalid_argument("dense::dynint::get_operators_and_interactions: dynamical interactions require an atom_diag "
                                        "with a single subspace, got " + std::to_string(ad.n_subspaces())
                                        + ". Build it with an empty list of conserved operators.");

        auto U_mat = ad.get_unitary_matrix(0);

        for (int i = 0; i < dynint_ops.size(); ++i) {
            auto op = dynint_ops[i];
            auto op_mat = U_mat * triqs_xca::atom_diag::get_op_mat(ad, op).block_mat[0] * nda::conj(nda::transpose(U_mat));
            Fs_ext(n_hyb + i, _, _) = op_mat;
            Fdags_ext(n_hyb + i, _, _) = nda::conj(nda::transpose(op_mat));
            //std::cout << "i = " << i << ", op = " << op << std::endl;
            //std::cout << "op_mat =" << op_mat << std::endl;
        }

        /*
        for(int i = 0; i < n_ext; ++i) {
            std::cout << "Operator (ext) " << i << ":\n";
            std::cout << "F_ext =" << Fs_ext(i, _, _) << "\n";
            std::cout << "Fdag_ext =" << Fdags_ext(i, _, _) << "\n";
        }

        for(int i = 0; i < n_ext; ++i) {
            for(int j = 0; j < n_ext; ++j) {
                std::cout << "Coefficients for operators " << i << " and " << j << ":";
                std::cout << ext_coeffs(_, i, j) << "\n";
            }
        }
        */
        
        return FSet{Fs_ext, Fdags_ext, ext_coeffs};
    }

    FSet get_operators_and_interactions(
        const triqs_atom_diag_t<true> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {
        return get_operators_and_interactions_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

    FSet get_operators_and_interactions(
        const triqs_atom_diag_t<false> &ad, 
        nda::array_const_view<dcomplex, 3> hyb_coeffs, 
        nda::array_const_view<dcomplex, 3> dynint_coeffs,  
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {
        return get_operators_and_interactions_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

}
