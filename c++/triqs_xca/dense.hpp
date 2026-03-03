#pragma once
#include <nda/nda.hpp>

namespace triqs_xca::dense {

    using nda::dcomplex;

    /**
     * @class DenseFSet
     * @brief Container for (linear combinations of) creation and annihilation operators in dense storage
     */
    class DenseFSet {
      public:
      nda::array<dcomplex, 3> Fs;
      nda::array<dcomplex, 3> F_dags;
      nda::array<dcomplex, 4> F_dag_bars;
      nda::array<dcomplex, 4> F_bars_refl;

      /**
       * @brief Constructor for DenseFSet
       * @param[in] Fs annihilation operators
       * @param[in] F_dags creation operators
       * @param[in] hyb_coeffs DLR coefficients of hybridization
       */
      DenseFSet(nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs);

      std::size_t get_num_orb_inds() const;
    };

} // namespace triqs_xca::dense