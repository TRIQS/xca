#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <triqs/operators/many_body_operator.hpp>

namespace triqs_xca {

  /**
   * @brief Is this many-body operator fermionic, i.e. of odd fermion parity?
   *
   * @details Fermion parity is the length of the operator's monomials modulo 2: odd-length monomials (c, c^dag, c^dag c^dag c, ...) are
   * fermionic, even-length ones (n, S_z, S+, S-, the identity, ...) are bosonic. Commuting is a different property, a commutator test
   * calls (S+, S-) fermionic since [S+, S-] = 2 S_z != 0, and (c, c) bosonic since [c, c] = 0.
   *
   * @param op the operator to classify
   * @param ctx caller name, used to prefix the exception messages
   * @throws std::runtime_error if op has monomials of both parities, or is identically zero
   * @return true if fermionic (odd), false if bosonic (even)
   */
  inline bool is_fermionic_parity(triqs::operators::many_body_operator_complex const &op, std::string const &ctx) {
    std::optional<bool> odd;
    for (auto const &term : op) {
      bool this_odd = (term.monomial.size() % 2 == 1);
      if (!odd)
        odd = this_odd;
      else if (*odd != this_odd)
        // an operator of mixed parity has no statistics, while c + c^dag c^dag c is all-odd and well defined
        throw std::runtime_error(ctx + ": operator has mixed fermion parity (both even- and odd-length monomials), "
                                       "so its statistics is undefined");
    }
    // no terms at all means the zero operator, the identity has one empty monomial of even length
    if (!odd) throw std::runtime_error(ctx + ": operator is identically zero, so its statistics is undefined");
    return *odd;
  }

  /**
   * @brief Classify every operator of a correlator and require them all to agree
   *
   * @details The parity is a property of each operator, so a mismatch such as (c_up, n_up) is detected.
   *
   * @param ops_tau operators at time tau
   * @param ops_0 operators at time 0
   * @param ctx caller name, used to prefix the exception messages
   * @throws std::runtime_error if the operators do not all share one fermion parity
   * @return the common statistics, true if fermionic
   */
  inline bool correlator_statistics(std::vector<triqs::operators::many_body_operator_complex> const &ops_tau,
                                    std::vector<triqs::operators::many_body_operator_complex> const &ops_0, std::string const &ctx) {
    std::optional<bool> common;
    auto consider = [&](triqs::operators::many_body_operator_complex const &op) {
      bool f = is_fermionic_parity(op, ctx);
      if (!common)
        common = f;
      else if (*common != f)
        throw std::runtime_error(ctx + ": All operators must have the same statistics (either all fermionic or all bosonic).");
    };
    for (auto const &op : ops_tau) consider(op);
    for (auto const &op : ops_0) consider(op);
    if (!common) throw std::runtime_error(ctx + ": no operators given, so the statistics is undefined");
    return *common;
  }

} // namespace triqs_xca
