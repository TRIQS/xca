#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/backbone.hpp>
#include <triqs_soehyb/dense_backbone.hpp>

TEST(DenseBackbone, one_vertex_and_edge) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-6;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // create cre/ann operators
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    // initialize backbone
    auto B = Backbone(topology, n);

    // set line directions
    nda::vector<int> fb = {1, fb1, 0};
    B.set_directions(fb);

    // set pole indices
    nda::vector<int> pole_inds = {0, r - 1};
    B.set_pole_inds(pole_inds, dlr_rf);

    // set orbital indices
    nda::vector<int> orb_inds = {1, 0, 1, 2, 0, 2};
    B.set_orb_inds(orb_inds);

    // multiply T by vertex 1
    for (int t = 0; t < r; t++) D.T(t, _, _) = nda::eye<dcomplex>(N);
    D.multiply_vertex_dense(B, 1);
    std::cout << B << std::endl;

    // do the same multiplication manually
    nda::array<dcomplex, 3> Tact(r, N, N);
    if (fb1 == 1) {
      for (int t = 0; t < r; t++) Tact(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Fs_dense(0, _, _);
    } else {
      for (int t = 0; t < r; t++) Tact(t, _, _) = F_dags_dense(0, _, _);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);

    // check that convolution with function on first edge is correct
    D.compose_with_edge_dense(B, 1);
    if (fb1 == 1) {
      Tact = itops.convolve(beta, Fermion, itops.vals2coefs(Gt_dense), itops.vals2coefs(Tact), TIME_ORDERED);
    } else {
      nda::array<dcomplex, 3> GKt_act(r, N, N);
      for (int t = 0; t < r; t++) GKt_act(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Gt_dense(t, _, _);
      Tact = itops.convolve(beta, Fermion, itops.vals2coefs(GKt_act), itops.vals2coefs(Tact), TIME_ORDERED);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);
  }
}

TEST(DenseBackbone, OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // load in functions from two_band.py
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-Deltat);
  auto hyb_refl_coeffs = nda::make_regular(-hyb_coeffs); // itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto D                      = DiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);

  // evaluate OCA self-energy contribution
  D.eval_diagram_dense(B);
  auto OCA_result = D.Sigma;

  // compare against manually-computed OCA result
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), eps);
}

TEST(DenseBackbone, PYTHON_OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-10;

  // load in functions from two_band.py
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat);       // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-Deltat);     // nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = nda::make_regular(-hyb_coeffs); // itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto D                      = DiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);

  // evaluate OCA self-energy contribution
  D.eval_diagram_dense(B);
  auto OCA_result = D.Sigma;

  int r = itops.rank(), N = 16;
  h5::file hfile("../test/c++/h5/two_band_py_Lambda10.h5", 'r');
  h5::group hgroup(hfile);
  nda::array<dcomplex, 3> NCA_py(r, N, N), OCA_py(r, N, N);
  h5::read(hgroup, "NCA", NCA_py);
  h5::read(hgroup, "OCA", OCA_py);
  OCA_py = OCA_py - NCA_py;

  std::cout << "OCA result: " << OCA_result(10, _, _) << std::endl;
  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }
  std::cout << "OCA py result: " << OCA_py_perm(10, _, _) << std::endl;

  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_py_perm)), eps);
}

TEST(DenseBackbone, third_order_manual) {
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute hybridization function
  auto hyb             = Deltat;
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute self-energy contribution of one third-order diagram topology,
  // with all forward hybridization lines and particular poles
  auto Sigma_manual           = third_order_dense_partial(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  auto B                      = Backbone(topology, n);
  nda::vector<int> fb{1, 1, 1}, pole_inds{7, 9};
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);

  // perform the same calculation using the a routine called by eval_diagram_dense()
  nda::array<dcomplex, 3> T(r, N, N), GKt(r, N, N), Tmu(r, N, N), Sigma_generic(r, N, N);
  nda::array<dcomplex, 4> Tkaps(n, r, N, N);
  nda::vector<int> states(6);
  // f_ix = o_ix + n^(m-1) * p_ix + (n * r)^(m-1) * fb_ix
  int pow_n_mm1  = static_cast<int>(std::pow(n, B.m - 1));
  int pow_nr_mm1 = static_cast<int>(std::pow(n * r, B.m - 1));
  int f_ix_start = pow_n_mm1 * (pole_inds(0) + r * pole_inds(1)) + pow_nr_mm1 * (fb(0) + 2 * fb(1) + 4 * fb(2));
  for (int f_ix_off = 0; f_ix_off < pow_n_mm1; f_ix_off++) {
    B.set_flat_index(f_ix_start + f_ix_off, dlr_rf);
    D.eval_backbone_fixed_indices_dense(B);
    B.reset_all_inds();
  }

  ASSERT_LE(nda::max_element(nda::abs(Sigma_manual(10, _, _) - D.Sigma(10, _, _))), eps);
}

TEST(DenseBackbone, PYTHON_third_order) {
  nda::array<int, 3> topologies = {{{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 4}, {2, 5}}};
  nda::vector<int> topo_sign{1, 1, 1, -1}; // topo_sign(i) = (-1)^{# of line crossings in topology i}

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute Fbars and Fdagbars
  auto hyb_coeffs      = itops.vals2coefs(Deltat);       // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-Deltat);     // nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = nda::make_regular(-hyb_coeffs); // itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute NCA and OCA
  auto NCA_result          = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> T_OCA = {{0, 2}, {1, 3}};
  auto B_OCA               = Backbone(T_OCA, n);
  // auto D                   = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, Gt_dense, Fset); // create DiagramEvaluator object
  auto D = DiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);
  D.eval_diagram_dense(B_OCA); // evaluate OCA diagram
  auto OCA_result = D.Sigma;   // get the result from the DiagramEvaluator
  D.reset();

  // arrays for storing results from third-order diagram computations
  auto third_order_result      = nda::zeros<dcomplex>(r, N, N);
  auto third_order_02_result   = nda::zeros<dcomplex>(r, N, N);
  auto third_order_0314_result = nda::zeros<dcomplex>(r, N, N);
  auto third_order_0315_result = nda::zeros<dcomplex>(r, N, N);
  auto third_order_04_result   = nda::zeros<dcomplex>(r, N, N);

  // compute third-order diagrams using generic backbone evaluators
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 4; i++) {
    auto B = Backbone(topologies(i, _, _), n); // create Backbone object for topology i
    D.eval_diagram_dense(B);
    auto eval = D.Sigma;
    third_order_result += topo_sign(i) * D.Sigma;
    if (i == 0)
      third_order_02_result = eval;
    else if (i == 1)
      third_order_0315_result = eval;
    else if (i == 2)
      third_order_04_result = eval;
    else
      third_order_0314_result = eval;
    D.reset();
  }
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Elapsed time for dense comp'n of 3rd order diags = " << duration.count() << " seconds" << std::endl;

  // load results from a run of twoband.py
  h5::file hfile("../test/c++/h5/two_band_py_Lambda10.h5", 'r');
  h5::group hgroup(hfile);
  nda::array<dcomplex, 3> NCA_py(r, N, N), OCA_py(r, N, N);
  nda::array<dcomplex, 3> third_order_py(r, N, N);
  nda::array<dcomplex, 3> third_order_py_02(r, N, N);
  nda::array<dcomplex, 3> third_order_py_0314(r, N, N);
  nda::array<dcomplex, 3> third_order_py_0315(r, N, N);
  nda::array<dcomplex, 3> third_order_py_04(r, N, N);
  h5::read(hgroup, "NCA", NCA_py);
  h5::read(hgroup, "OCA", OCA_py);
  OCA_py = OCA_py - NCA_py;

  h5::read(hgroup, "third_order", third_order_py);
  third_order_py = -third_order_py + OCA_py + NCA_py;
  h5::read(hgroup, "third_order_[(0, 2), (1, 4), (3, 5)]", third_order_py_02);
  h5::read(hgroup, "third_order_[(0, 3), (1, 4), (2, 5)]", third_order_py_0314);
  h5::read(hgroup, "third_order_[(0, 3), (1, 5), (2, 4)]", third_order_py_0315);
  h5::read(hgroup, "third_order_[(0, 4), (1, 3), (2, 5)]", third_order_py_04);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm              = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm              = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_perm      = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_02_perm   = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_0314_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_0315_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto third_order_py_04_perm   = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j)              = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j)              = OCA_py(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_perm(t, i, j)      = third_order_py(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_02_perm(t, i, j)   = third_order_py_02(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_0314_perm(t, i, j) = third_order_py_0314(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_0315_perm(t, i, j) = third_order_py_0315(t, fock_state_order[i], fock_state_order[j]);
        third_order_py_04_perm(t, i, j)   = third_order_py_04(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  ASSERT_LE(nda::max_element(nda::abs(NCA_result - NCA_py_perm)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_py_perm)), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_02_result - third_order_py_02_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_0314_result - third_order_py_0314_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_0315_result - third_order_py_0315_perm)), 100 * eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_04_result - third_order_py_04_perm)), 100 * eps);

  ASSERT_LE(nda::max_element(nda::abs(third_order_result - third_order_py_perm)), 100 * eps);
}

TEST(DenseBackbone, PYTHON_OCA_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // call two band helper just for Gt_dense, Fs_dense, F_dags_dense
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  int p = 7;
  int n = 4;
  nda::array<dcomplex, 3> hyb(r, n, n), hyb_coeffs(p, n, n);
  hyb           = {{{-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105}},
                   {{-0.4867352379479528, -0.4867352379479528, -0., -0.},
                    {-0.4867352379479528, -0.4867352379479528, -0., -0.},
                    {-0., -0., -0.4867352379479528, -0.4867352379479528},
                    {-0., -0., -0.4867352379479528, -0.4867352379479528}},
                   {{-0.4603465101833711, -0.4603465101833711, -0., -0.},
                    {-0.4603465101833711, -0.4603465101833711, -0., -0.},
                    {-0., -0., -0.4603465101833711, -0.4603465101833711},
                    {-0., -0., -0.4603465101833711, -0.4603465101833711}},
                   {{-0.4239204950540695, -0.4239204950540695, -0., -0.},
                    {-0.4239204950540695, -0.4239204950540695, -0., -0.},
                    {-0., -0., -0.4239204950540695, -0.4239204950540695},
                    {-0., -0., -0.4239204950540695, -0.4239204950540695}},
                   {{-0.3716597467714097, -0.3716597467714097, -0., -0.},
                    {-0.3716597467714097, -0.3716597467714097, -0., -0.},
                    {-0., -0., -0.3716597467714097, -0.3716597467714097},
                    {-0., -0., -0.3716597467714097, -0.3716597467714097}},
                   {{-0.2884886574148449, -0.2884886574148449, -0., -0.},
                    {-0.2884886574148449, -0.2884886574148449, -0., -0.},
                    {-0., -0., -0.2884886574148449, -0.2884886574148449},
                    {-0., -0., -0.2884886574148449, -0.2884886574148449}},
                   {{-0.2479810727230272, -0.2479810727230272, -0., -0.},
                    {-0.2479810727230272, -0.2479810727230272, -0., -0.},
                    {-0., -0., -0.2479810727230272, -0.2479810727230272},
                    {-0., -0., -0.2479810727230272, -0.2479810727230272}},
                   {{-0.2065525284769785, -0.2065525284769785, -0., -0.},
                    {-0.2065525284769785, -0.2065525284769785, -0., -0.},
                    {-0., -0., -0.2065525284769785, -0.2065525284769785},
                    {-0., -0., -0.2065525284769785, -0.2065525284769785}},
                   {{-0.1635819676241178, -0.1635819676241178, -0., -0.},
                    {-0.1635819676241178, -0.1635819676241178, -0., -0.},
                    {-0., -0., -0.1635819676241178, -0.1635819676241178},
                    {-0., -0., -0.1635819676241178, -0.1635819676241178}},
                   {{-0.1326995066858671, -0.1326995066858671, -0., -0.},
                    {-0.1326995066858671, -0.1326995066858671, -0., -0.},
                    {-0., -0., -0.1326995066858671, -0.1326995066858671},
                    {-0., -0., -0.1326995066858671, -0.1326995066858671}},
                   {{-0.1225444804140666, -0.1225444804140666, -0., -0.},
                    {-0.1225444804140666, -0.1225444804140666, -0., -0.},
                    {-0., -0., -0.1225444804140666, -0.1225444804140666},
                    {-0., -0., -0.1225444804140666, -0.1225444804140666}},
                   {{-0.1282199855712255, -0.1282199855712255, -0., -0.},
                    {-0.1282199855712255, -0.1282199855712255, -0., -0.},
                    {-0., -0., -0.1282199855712255, -0.1282199855712255},
                    {-0., -0., -0.1282199855712255, -0.1282199855712255}},
                   {{-0.1386184647087601, -0.1386184647087601, -0., -0.},
                    {-0.1386184647087601, -0.1386184647087601, -0., -0.},
                    {-0., -0., -0.1386184647087601, -0.1386184647087601},
                    {-0., -0., -0.1386184647087601, -0.1386184647087601}},
                   {{-0.1720919948804938, -0.1720919948804938, -0., -0.},
                    {-0.1720919948804938, -0.1720919948804938, -0., -0.},
                    {-0., -0., -0.1720919948804938, -0.1720919948804938},
                    {-0., -0., -0.1720919948804938, -0.1720919948804938}},
                   {{-0.2300400167898313, -0.2300400167898313, -0., -0.},
                    {-0.2300400167898313, -0.2300400167898313, -0., -0.},
                    {-0., -0., -0.2300400167898313, -0.2300400167898313},
                    {-0., -0., -0.2300400167898313, -0.2300400167898313}},
                   {{-0.3000508284935615, -0.3000508284935615, -0., -0.},
                    {-0.3000508284935615, -0.3000508284935615, -0., -0.},
                    {-0., -0., -0.3000508284935615, -0.3000508284935615},
                    {-0., -0., -0.3000508284935615, -0.3000508284935615}},
                   {{-0.3759657450111002, -0.3759657450111002, -0., -0.},
                    {-0.3759657450111002, -0.3759657450111002, -0., -0.},
                    {-0., -0., -0.3759657450111002, -0.3759657450111002},
                    {-0., -0., -0.3759657450111002, -0.3759657450111002}},
                   {{-0.4545389745912252, -0.4545389745912252, -0., -0.},
                    {-0.4545389745912252, -0.4545389745912252, -0., -0.},
                    {-0., -0., -0.4545389745912252, -0.4545389745912252},
                    {-0., -0., -0.4545389745912252, -0.4545389745912252}},
                   {{-0.4821599768174421, -0.4821599768174421, -0., -0.},
                    {-0.4821599768174421, -0.4821599768174421, -0., -0.},
                    {-0., -0., -0.4821599768174421, -0.4821599768174421},
                    {-0., -0., -0.4821599768174421, -0.4821599768174421}},
                   {{-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105}}};
  hyb_coeffs    = {{{0.0028042961182163, 0.0028042961182163, 0., 0.},
                    {0.0028042961182163, 0.0028042961182163, 0., 0.},
                    {0., 0., 0.0028042961182163, 0.0028042961182163},
                    {0., 0., 0.0028042961182163, 0.0028042961182163}},
                   {{0.088487039172428, 0.088487039172428, 0., 0.},
                    {0.088487039172428, 0.088487039172428, 0., 0.},
                    {0., 0., 0.088487039172428, 0.088487039172428},
                    {0., 0., 0.088487039172428, 0.088487039172428}},
                   {{0.1575418229076625, 0.1575418229076625, 0., 0.},
                    {0.1575418229076625, 0.1575418229076625, 0., 0.},
                    {0., 0., 0.1575418229076625, 0.1575418229076625},
                    {0., 0., 0.1575418229076625, 0.1575418229076625}},
                   {{0.1953880665937937, 0.1953880665937937, 0., 0.},
                    {0.1953880665937937, 0.1953880665937937, 0., 0.},
                    {0., 0., 0.1953880665937937, 0.1953880665937937},
                    {0., 0., 0.1953880665937937, 0.1953880665937937}},
                   {{0.2145207908265103, 0.2145207908265103, 0., 0.},
                    {0.2145207908265103, 0.2145207908265103, 0., 0.},
                    {0., 0., 0.2145207908265103, 0.2145207908265103},
                    {0., 0., 0.2145207908265103, 0.2145207908265103}},
                   {{0.1832496441339733, 0.1832496441339733, 0., 0.},
                    {0.1832496441339733, 0.1832496441339733, 0., 0.},
                    {0., 0., 0.1832496441339733, 0.1832496441339733},
                    {0., 0., 0.1832496441339733, 0.1832496441339733}},
                   {{0.1580088741667851, 0.1580088741667851, 0., 0.},
                    {0.1580088741667851, 0.1580088741667851, 0., 0.},
                    {0., 0., 0.1580088741667851, 0.1580088741667851},
                    {0., 0., 0.1580088741667851, 0.1580088741667851}}};
  auto hyb_refl = nda::make_regular(-hyb);
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = nda::make_regular(-hyb_coeffs);

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

  auto OCA_dense_result = OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // load NCA and OCA results from twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py_semic.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that dense OCA calculation agree with twoband.py
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_py_perm + NCA_py_perm)), eps);

  // generic diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto Fset                   = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);
  auto D                      = DiagramEvaluator(beta, itops, hyb, hyb_refl, hyb_poles, Gt_dense, Fset);
  D.eval_diagram_dense(B);   // evaluate OCA diagram
  auto OCA_result = D.Sigma; // get the result from the DiagramEvaluator

  // compare with the dense result
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), eps);
}