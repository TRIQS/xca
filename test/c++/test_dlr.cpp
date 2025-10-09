/*******************************************************************************
 *
 * triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, Z. Huang
 *
 * triqs_soehyb is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "triqs_soehyb/strong_cpl.hpp"
#include "nda/nda.hpp"
#include <cppdlr/cppdlr.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <functional>
#include <gtest/gtest.h>
#include <nda/algorithms.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout/policies.hpp>
#include <chrono>

using namespace cppdlr;
using namespace nda;
void construct_G_and_Delta(nda::array_view<dcomplex, 3> Gt, nda::array_view<dcomplex, 3> Deltat, nda::vector_const_view<double> dlr_it_actual,
                           double beta, double alpha_1, double alpha_2, int r);
nda::array<dcomplex, 3> OCAtrue(double alpha_1, double alpha_2, double beta, nda::vector_const_view<double> dlr_it_actual, int r, int N,
                                int dim);
nda::array<dcomplex, 3> Diagramtrue_3rd(double alpha_1, double alpha_2, double beta, nda::vector_const_view<double> dlr_it_actual, int r, int N,
                                        int dim);

TEST(strong_coupling, exponential_functions) {

  // --- Problem setup --- //

  // Set problem parameters
  double beta   = 2; // Inverse temperature
  const int N   = 2; // Dimension of Greens function. Do not change, or you'll have to rewrite G(t)
  const int dim = 3; //Dimension of hybridization. One can change dim as they want.

  // Set DLR parameters
  double lambda      = beta * 5;
  double eps         = 1.0e-12;
  auto dlr_rf        = build_dlr_rf(lambda, eps);  // Get DLR frequencies
  auto itops         = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  nda::vector<double> dlr_it_actual = dlr_it;
  for (int i = 0; i < r; ++i) {
    if (dlr_it(i) < 0) dlr_it_actual(i) = dlr_it(i) + 1;
  }

  //parameters in exponential functions we will use
  double alpha_1 = 0.5;
  double alpha_2 = 0.1;
  //construct G(t) = [0, exp(-alpha_1*t); exp(-alpha_1*t) 0], Deltat = exp(-alpha2 * t)
  auto Gt     = nda::array<dcomplex, 3>(r, N, N);
  auto Deltat = nda::array<dcomplex, 3>(r, dim, dim);
  construct_G_and_Delta(Gt, Deltat, dlr_it_actual, beta, alpha_1, alpha_2, r);

  //construct Gdlr and Delta dlr
  auto Gdlr     = itops.vals2coefs(Gt);
  auto Deltadlr = itops.vals2coefs(Deltat);
  //reflect Deltat
  auto Deltat_reflect   = itops.reflect(Deltat);
  auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);

  //decomposition of hybridization
  auto Delta_decomp = hyb_decomp(Deltadlr, dlr_rf);
  Delta_decomp.check_accuracy(Deltat, dlr_it);
  auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf);
  Delta_decomp_reflect.check_accuracy(Deltat_reflect, dlr_it);
  //F matrices
  auto F = nda::array<dcomplex, 3>(dim, N, N);

  for (int i = 0; i < dim; ++i) F(i, _, _) = eye<dcomplex>(N);
  auto F_dag = F;

  //construct U_tilde, V_tilde, c
  hyb_F Delta_F(N, r, dim);
  hyb_F Delta_F_reflect(N, r, dim);

  Delta_F.update_inplace(Delta_decomp, dlr_it, F, F_dag);
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dag, F);

  bool backward = false;
  auto fb2      = nda::vector<int64_t>(2);
  fb2           = 0;
  auto fb3      = nda::vector<int64_t>(3);
  fb3           = 0;
  //Test OCA(2nd order) diagram
  std::cout << "Testing OCA diagram..." << std::endl;
  auto OCAdiagram = Sigma_OCA_calc(Delta_F, Deltat, Deltat, Gt, itops, beta, F, F, backward);
  auto D2         = nda::array<int, 2>{{0, 2}, {1, 3}}; // Diagram topology

  auto begin       = std::chrono::high_resolution_clock::now();
  auto OCAdiagram2 = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, Deltat, Deltat, Gt, itops, beta, F, F, fb2, backward);
  auto end         = std::chrono::high_resolution_clock::now();
  auto elapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t2        = elapsed.count();

  std::cout << "Difference between OCA_calc and Diagram_calc for OCA diagram is " << max_element(abs(OCAdiagram - OCAdiagram2)) << std::endl;
  auto OCA_true = OCAtrue(alpha_1, alpha_2, beta, dlr_it_actual, r, N, dim);
  std::cout << "Error of OCA_Diagram is " << max_element(abs(OCAdiagram - OCA_true)) << std::endl;
  //EXPECT_LT(max_element(abs(OCAdiagram - OCA_true)), 1e-12);

  //Test third order diagram
  std::cout << "Testing third order diagram... " << std::endl;
  auto D3                     = nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}};
  begin                       = std::chrono::high_resolution_clock::now();
  auto diagram_3rd_order      = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D3, Deltat, Deltat, Gt, itops, beta, F, F, fb3, backward);
  end                         = std::chrono::high_resolution_clock::now();
  elapsed                     = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  double t3                   = elapsed.count();
  auto diagram_3rd_order_true = Diagramtrue_3rd(alpha_1, alpha_2, beta, dlr_it_actual, r, N, dim);
  std::cout << "Error of 3rd order diagram is " << max_element(abs(diagram_3rd_order_true - diagram_3rd_order)) << std::endl;
  int P = Delta_F.w.shape(0);
  std::cout << "r here is " << r << ", P here is " << P << ". The theoretical complexity is P^(m-1)*(m*r^2+n*r)" << std::endl;
  std::cout << "Time spent for m=2, 3 diagram is " << t2 / 1000 << ", " << t3 / 1000 << " seconds" << std::endl;
  // auto OCA_all = Diagram_calc_sum_all(Delta_F,Delta_F_reflect,D3, Deltat, Deltat_reflect, Gt,itops, beta, F,  F_dag);
}

// TEST(strong_coupling, high_order_diagrams) {
//     // Set problem parameters
//     double beta = 2; // Inverse temperature
//     const int N = 2;     // Dimension of Greens function. Do not change, or you'll have to rewrite G(t)
//     int Num = 128;   // Size of equidist grid
//     const int dim = 3; //Dimension of hybridization. One can change dim as they want.

//     // Set DLR parameters
//     double lambda = beta*5;
//     double eps = 1.0e-12;
//     auto dlr_rf = build_dlr_rf(lambda, eps); // Get DLR frequencies
//     auto itops = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
//     auto const & dlr_it = itops.get_itnodes();
//     int r = itops.rank();

//     nda::vector<double> dlr_it_actual = dlr_it;
//     for (int i = 0;i<r;++i) {if (dlr_it(i)<0) dlr_it_actual(i) = dlr_it(i)+1; }

//     //parameters in exponential functions we will use
//     double alpha_1 = 0.5;
//     //construct G(t) = [0, exp(-alpha_1*t); exp(-alpha_1*t) 0], Deltat = exp(-alpha2 * t)
//     auto Gt = nda::array<dcomplex, 3>(r, N, N);
//     auto Deltat = nda::array<dcomplex, 3>(r, dim, dim);
//     construct_G_and_Delta(Gt,  Deltat, dlr_it_actual, beta, alpha_1, 0.0, r);

//     //construct Gdlr and Delta dlr
//     auto Gdlr = itops.vals2coefs(Gt);
//     auto Deltadlr = itops.vals2coefs(Deltat);

//     //reflect Deltat
//     auto Deltat_reflect  = itops.reflect(Deltat);
//     auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);

//     auto F = nda::array<dcomplex,3>(dim,N,N);
//     for (int i = 0; i<dim;++i) F(i,_,_) = eye<dcomplex>(N);

//     // Diagram topology that we will try
//     auto D2 = nda::array<int,2>{{0,2},{1,3}};
//     auto D3 = nda::array<int,2>{{0,2},{1,4},{3,5}};
//     auto D4= nda::array<int,2>{{0,2},{1,6},{3,5},{4,7}};
//     auto D5= nda::array<int,2>{{0,2},{1,8},{3,6},{4,9},{5,7}};
//     auto D6 = nda::array<int,2>{{0,2},{1,7},{3,9},{4,10},{5,8},{6,11}};
//     auto D7 = nda::array<int,2>{{0,2},{1,7},{3,13},{4,10},{5,8},{6,11},{9,12}};
//     auto D8 = nda::array<int,2>{{0,2},{1,14},{3,13},{4,10},{5,8},{6,11},{9,12},{7,15}};

//     //construct true solutions
//     auto OCA_true = nda::array<dcomplex,3>(Gt.shape());
//     OCA_true = 0;
//     for (int i=0;i<r;++i) OCA_true(i,0,1) =exp(-(alpha_1)*beta*dlr_it_actual(i))*(beta*dlr_it_actual(i)/1.0)*(beta*dlr_it_actual(i)/2.0);
//     OCA_true(_,1,0) = OCA_true(_,0,1);
//     OCA_true = OCA_true * pow(dim,2);

//     auto diagram_3rd_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_3rd_order_true(i,_,_) = OCA_true(i,_,_)*(beta*dlr_it_actual(i)/3.0)*(beta*dlr_it_actual(i)/4.0)*dim;

//     auto diagram_4th_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_4th_order_true(i,_,_) =diagram_3rd_order_true(i,_,_) *(beta*dlr_it_actual(i)/5.0)*(beta*dlr_it_actual(i)/6.0)* dim;

//     auto diagram_5th_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_5th_order_true(i,_,_) = diagram_4th_order_true(i,_,_)*(beta*dlr_it_actual(i)/7.0) * (beta*dlr_it_actual(i)/8.0)* dim;

//     auto diagram_6th_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_6th_order_true(i,_,_) = diagram_5th_order_true(i,_,_)*(beta/9.0)*dlr_it_actual(i) * (beta/10.0)*dlr_it_actual(i)* dim;

//     auto diagram_7th_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_7th_order_true(i,_,_) = diagram_6th_order_true(i,_,_)*(beta/11.0)*dlr_it_actual(i) * (beta/12.0)*dlr_it_actual(i)*dim;

//     auto diagram_8th_order_true = nda::array<dcomplex,3>(Gt.shape());
//     for (int i=0;i<r;++i) diagram_8th_order_true(i,_,_) = diagram_7th_order_true(i,_,_)*(beta*dlr_it_actual(i)/13.0) * (beta*dlr_it_actual(i)/14.0) * dim;

//     std::cout<<"Testing high order diagrams:"<<std::endl;
//     nda::array<double,1> pol(1);
//     pol(0) =  0.0*beta;
//     nda::array<dcomplex,3> A(1,dim,dim);
//     A(0,_,_) = eye<dcomplex>(dim)/k_it(0,0.0*beta);

//     auto Delta_decomp_simple = hyb_decomp(A,pol);
//     Delta_decomp_simple.check_accuracy(Deltat, dlr_it);
//     bool backward = false;
//     auto fb2 =  nda::vector<int>(2); fb2=0;
//     auto fb3 =  nda::vector<int>(3); fb3=0;
//     auto fb4 =  nda::vector<int>(4); fb4=0;
//     auto fb5 =  nda::vector<int>(5); fb5=0;
//     auto fb6 =  nda::vector<int>(6); fb6=0;
//     auto fb7 =  nda::vector<int>(7); fb7=0;
//     auto fb8 =  nda::vector<int>(8); fb8=0;

//     auto Delta_F_simple = hyb_F(Delta_decomp_simple, dlr_rf, dlr_it, F, F);
//     //calculating diagrams
//     auto begin = std::chrono::high_resolution_clock::now();
//     auto OCAdiagram_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D2,Deltat,Deltat, Gt,itops,beta, F,  F, fb2,backward);
//     auto end = std::chrono::high_resolution_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t2_simple =  elapsed.count();
//     std::cout<< "Error of OCA_Diagram is "<< max_element(abs(OCAdiagram_simple - OCA_true))<<"; maximum of diagram is "<< max_element(abs(OCA_true)) <<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_3rd_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D3,Deltat,Deltat, Gt,itops,beta, F,  F,fb3,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t3_simple =  elapsed.count();
//     std::cout<< "Error of 3rd order Diagram is "<< max_element(abs(diagram_3rd_order_simple - diagram_3rd_order_true)) <<"; maximum of diagram is "<< max_element(abs(diagram_3rd_order_true))<<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_4th_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D4,Deltat,Deltat, Gt,itops,beta, F,  F,fb4,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t4_simple =  elapsed.count();
//     //std::cout<<"Difference of 4th order diagram between two methods is "<<max_element(abs(diagram_4th_order_simple - diagram_4th_order)) <<std::endl;
//     std::cout<< "Error of 4th order Diagram is "<< max_element(abs(diagram_4th_order_simple - diagram_4th_order_true)) <<"; maximum of diagram is "<< max_element(abs(diagram_4th_order_true))<<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_5th_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D5,Deltat,Deltat, Gt,itops,beta, F,  F,fb5,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t5_simple =  elapsed.count();
//     std::cout<< "Error of 5th order Diagram is "<< max_element(abs(diagram_5th_order_simple - diagram_5th_order_true))<<"; maximum of diagram is "<< max_element(abs(diagram_5th_order_true)) <<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_6th_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D6,Deltat,Deltat, Gt,itops,beta, F,  F,fb6,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t6_simple =  elapsed.count();
//     std::cout<< "Error of 6th order Diagram is "<< max_element(abs(diagram_6th_order_simple - diagram_6th_order_true))<<"; maximum of diagram is "<< max_element(abs(diagram_6th_order_true)) <<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_7th_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D7,Deltat,Deltat, Gt,itops,beta, F,  F,fb7,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t7_simple =  elapsed.count();
//     std::cout<< "Error of 7th order Diagram is "<< max_element(abs(diagram_7th_order_simple - diagram_7th_order_true)) <<"; maximum of diagram is "<< max_element(abs(diagram_7th_order_true))<<std::endl;

//     begin = std::chrono::high_resolution_clock::now();
//     auto diagram_8th_order_simple = Sigma_Diagram_calc(Delta_F_simple,Delta_F_simple,D8,Deltat,Deltat, Gt,itops,beta, F,  F,fb8,backward);
//     end = std::chrono::high_resolution_clock::now();
//     elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
//     double t8_simple =  elapsed.count();

//     std::cout<< "Error of 8th order Diagram is "<< max_element(abs(diagram_8th_order_simple - diagram_8th_order_true)) <<"; maximum of diagram is "<< max_element(abs(diagram_8th_order_true))<<std::endl;

//     std::cout<<"Time spent in seconds"<<std::endl;
//     std::cout<<"m=2      "<<t2_simple/1000<<std::endl;
//     std::cout<<"m=3      "<<t3_simple/1000<<std::endl;
//     std::cout<<"m=4      "<<t4_simple/1000<<std::endl;
//     std::cout<<"m=5      "<<t5_simple/1000<<std::endl;
//     std::cout<<"m=6      "<<t6_simple/1000<<std::endl;
//     std::cout<<"m=7      "<<t7_simple/1000<<std::endl;
//     std::cout<<"m=8      "<<t8_simple/1000<<std::endl;

// }

TEST(strong_coupling, G_diagrams) {
  // Set problem parameters
  double beta   = 1; // Inverse temperature
  const int N   = 1; // Dimension of Greens function. Do not change, or you'll have to rewrite G(t)
  const int dim = 1; //Dimension of hybridization. One can change dim as they want.

  // Set DLR parameters
  double lambda      = beta * 10;
  double eps         = 1.0e-12;
  auto dlr_rf        = build_dlr_rf(lambda, eps);  // Get DLR frequencies
  auto itops         = imtime_ops(lambda, dlr_rf); // Get DLR imaginary time object
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  nda::vector<double> dlr_it_actual = dlr_it;
  for (int i = 0; i < r; ++i) {
    if (dlr_it(i) < 0) dlr_it_actual(i) = dlr_it(i) + 1;
  }

  //parameters in exponential functions we will use
  double alpha_1 = 3;
  double alpha_2 = 0.0;
  //construct G(t) = [0, exp(-alpha_1*t); exp(-alpha_1*t) 0], Deltat = exp(-alpha2 * t)
  auto Gt     = nda::array<dcomplex, 3>(r, N, N);
  auto Deltat = nda::array<dcomplex, 3>(r, dim, dim);
  construct_G_and_Delta(Gt, Deltat, dlr_it_actual, beta, alpha_1, alpha_2, r);

  //construct Gdlr and Delta dlr
  auto Gdlr     = itops.vals2coefs(Gt);
  auto Deltadlr = itops.vals2coefs(Deltat);

  //reflect Deltat
  auto Deltat_reflect   = itops.reflect(Deltat);
  auto Deltadlr_reflect = itops.vals2coefs(Deltat_reflect);

  auto F = nda::array<dcomplex, 3>(dim, N, N);
  for (int i = 0; i < dim; ++i) F(i, _, _) = eye<dcomplex>(N);

  // Diagram topology that we will try
  auto D2 = nda::array<int, 2>{{0, 2}, {1, 3}};
  auto D3 = nda::array<int, 2>{{0, 2}, {1, 4}, {3, 5}};

  //construct true solutions
  auto G_OCA_true_00 = nda::array<dcomplex, 3>(Gt.shape(0), 1, 1);
  G_OCA_true_00      = 0;
  for (int i = 0; i < r; ++i) G_OCA_true_00(i, 0, 0) = beta * beta * dlr_it_actual(i) * (1 - dlr_it_actual(i)) * exp(-alpha_1 * beta) * dim * N;

  nda::array<double, 1> pol(1);
  pol(0) = 0.0 * beta;
  nda::array<dcomplex, 3> A(1, dim, dim);
  A(0, _, _) = eye<dcomplex>(dim) / k_it(0, 0.0 * beta);

  auto Delta_decomp_simple = hyb_decomp(A, pol);
  Delta_decomp_simple.check_accuracy(Deltat, dlr_it);
  auto fb2 = nda::vector<int64_t>(2);
  fb2      = 0;
  auto fb3 = nda::vector<int64_t>(3);
  fb3      = 0;

  hyb_F Delta_F_simple(N, r, dim);
  Delta_F_simple.update_inplace(Delta_decomp_simple, dlr_it, F, F);

  //calculating diagrams
  auto G_OCAdiagram_simple = G_Diagram_calc(Delta_F_simple, Delta_F_simple, D2, Gt, itops, beta, F, F, fb2);
  std::cout << G_OCAdiagram_simple(0, _, _);

  std::cout << max_element(abs(G_OCAdiagram_simple(_, 0, 0) - G_OCA_true_00(_, 0, 0)));
  auto G_OCAdiagram_simple_all = G_Diagram_calc_sum_all(Delta_F_simple, Delta_F_simple, D2, Gt, itops, beta, F, F);
}

void construct_G_and_Delta(nda::array_view<dcomplex, 3> Gt, nda::array_view<dcomplex, 3> Deltat, nda::vector_const_view<double> dlr_it_actual,
                           double beta, double alpha_1, double alpha_2, int r) {
  auto G_01 = exp(-alpha_1 * beta * dlr_it_actual);
  Gt        = 0;
  if (Gt.shape(1) == 1) {
    Gt(_, 0, 0) = G_01;
  } else {
    Gt(_, 0, 1) = G_01;
    Gt(_, 1, 0) = G_01;
  }
  Deltat = 0;
  for (int i = 0; i < r; ++i) Deltat(i, _, _) = exp(-alpha_2 * beta * dlr_it_actual(i)) * eye<dcomplex>(Deltat.shape(1));
}

nda::array<dcomplex, 3> OCAtrue(double alpha_1, double alpha_2, double beta, nda::vector_const_view<double> dlr_it_actual, int r, int N,
                                int dim) {
  auto OCA_true = nda::array<dcomplex, 3>(r, N, N);
  OCA_true      = 0;
  if (abs(alpha_2) > 1.0e-10) {
    for (int i = 0; i < r; ++i)
      OCA_true(i, 0, 1) = (exp(-(alpha_1 + alpha_2) * beta * dlr_it_actual(i)))
         * (beta * dlr_it_actual(i) + (exp(-alpha_2 * beta * dlr_it_actual(i)) - 1) / alpha_2) / alpha_2;
  } else {
    for (int i = 0; i < r; ++i)
      OCA_true(i, 0, 1) = exp(-(alpha_1)*beta * dlr_it_actual(i)) * (beta * dlr_it_actual(i) / 1.0) * (beta * dlr_it_actual(i) / 2.0);
  }
  OCA_true(_, 1, 0) = OCA_true(_, 0, 1);
  OCA_true          = OCA_true * pow(dim, 2);
  return OCA_true;
}
nda::array<dcomplex, 3> Diagramtrue_3rd(double alpha_1, double alpha_2, double beta, nda::vector_const_view<double> dlr_it_actual, int r, int N,
                                        int dim) {
  auto diagram_3rd_order_true = nda::array<dcomplex, 3>(r, N, N);

  diagram_3rd_order_true = 0;
  if (abs(alpha_2) > 1.0e-10) {
    for (int i = 0; i < r; ++i)
      diagram_3rd_order_true(i, 0, 1) = (1 / (2 * pow(alpha_2, 4))) * exp(-(alpha_1 + alpha_2) * beta * dlr_it_actual(i))
         * (pow(alpha_2, 2) * pow(beta * dlr_it_actual(i), 2) - 4 * alpha_2 * beta * dlr_it_actual(i)
            - 2 * exp(-alpha_2 * beta * dlr_it_actual(i)) * (alpha_2 * beta * dlr_it_actual(i) + 3) + 6);
  } else {
    for (int i = 0; i < r; ++i)
      diagram_3rd_order_true(i, 0, 1) = exp(-(alpha_1)*beta * dlr_it_actual(i)) * (beta * dlr_it_actual(i) / 1.0) * (beta * dlr_it_actual(i) / 2.0)
         * (beta * dlr_it_actual(i) / 3.0) * (beta * dlr_it_actual(i) / 4.0);
  }
  diagram_3rd_order_true(_, 1, 0) = diagram_3rd_order_true(_, 0, 1);
  diagram_3rd_order_true          = diagram_3rd_order_true * pow(dim, 3);
  return diagram_3rd_order_true;
}
