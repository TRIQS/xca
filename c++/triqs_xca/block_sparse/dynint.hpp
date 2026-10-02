#pragma once

#include <nda/nda.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/atom_diag_utils.hpp"
#include "triqs_xca/block_sparse/atom_diag_utils.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"


namespace triqs_xca::dynint {

    using triqs_xca::atom_diag::triqs_atom_diag_t;
    using triqs_xca::block_sparse::BlockOpSymQuartet;

    /**
     * @brief Block-sparse field operators extended by the dynamical-interaction operators
     *
     * The block-sparse analogue of get_operators_and_interactions_dense(). The fermionic operators are grouped
     * into symmetry sets as in atom_diag::get_operators(), and the interaction operators are grouped by identical
     * connection row and appended as further symmetry sets. The coefficients are extended block-diagonally by
     * hyb::get_extended_coefficients(). Unlike the dense path there is no restriction on the number of atom_diag
     * subspaces, an interaction operator only has to map each subspace to at most one target.
     *
     * @param[in] ad AtomDiag object
     * @param[in] hyb_coeffs Hybridization coefficients, shape (p, n_hyb, n_hyb)
     * @param[in] dynint_coeffs Dynamical-interaction coefficients, shape (p, n_int, n_int)
     * @param[in] dynint_ops The n_int interaction operators
     * @return The extended operator quartet and the symmetry-set label of each of the n_hyb + n_int flavours
     */
    std::tuple<BlockOpSymQuartet, nda::vector<long>> get_operators_and_interactions(
        const triqs_atom_diag_t<true> &ad,
        nda::array_const_view<dcomplex, 3> hyb_coeffs,
        nda::array_const_view<dcomplex, 3> dynint_coeffs,
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops);

    std::tuple<BlockOpSymQuartet, nda::vector<long>> get_operators_and_interactions(
        const triqs_atom_diag_t<false> &ad,
        nda::array_const_view<dcomplex, 3> hyb_coeffs,
        nda::array_const_view<dcomplex, 3> dynint_coeffs,
        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops);

}
