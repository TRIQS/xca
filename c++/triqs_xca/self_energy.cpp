#include <cppdlr/dlr_build.hpp>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include <triqs/mesh.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

using namespace triqs;
using namespace triqs::atom_diag;

std::vector<nda::array<dcomplex, 3>> compute_self_energy(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb,
                                                         nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                         triqs::atom_diag::atom_diag<false> ad, int order) {
  auto dlr_rf               = build_dlr_rf(Lambda, eps);
  auto itops                = imtime_ops(Lambda, dlr_rf);
  auto Gt                   = ad_to_atom_prop(ad, beta, itops);
  int norb2                 = static_cast<int>(hyb_coeffs.extent(1));
  int norb                  = norb2 / 2;
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs, hyb_coeffs);
  auto hyb_refl             = itops.reflect(hyb);
  auto Sigma_BDOF           = NCA_bs(hyb, hyb_refl, Gt, Fq);

  if (order == 1) { return Sigma_BDOF.get_blocks(); }

  DiagramEvaluator D(beta, itops, hyb, hyb_refl, hyb_poles, Gt, Fq);
  nda::array<int, 2> T_OCA = {{0, 2}, {1, 3}};
  auto B_OCA               = Backbone(T_OCA, norb);
  D.eval_self_energy(B_OCA);
  Sigma_BDOF += D.get_self_energy();

  if (order == 2) { return Sigma_BDOF.get_blocks(); }

  nda::array<int, 3> T_third = {{{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 4}, {2, 5}}};
  for (int i = 0; i < 4; ++i) {
    D.reset();
    auto B_third = Backbone(T_third(i, _, _), norb);
    D.eval_self_energy(B_third);
    if (i == 3) {
      Sigma_BDOF += -1 * D.get_self_energy();
    } else {
      Sigma_BDOF += D.get_self_energy();
    }
  }

  return Sigma_BDOF.get_blocks();
}