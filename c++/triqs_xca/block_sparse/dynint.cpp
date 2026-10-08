#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/block_sparse/dynint.hpp"

namespace triqs_xca::block_sparse::dynint {

    using cppdlr::_;

    using nda::range;

    namespace {

        /// One dynamical-interaction operator, resolved against the atom_diag subspaces and rotated to the Fock basis.
        struct DynintOpBlocks {
            nda::vector<int> connection;                     // connection row of O, -1 where the block is absent
            nda::vector<int> connection_dag;                 // connection row of O^dag, the inverse map
            std::vector<nda::array<dcomplex, 2>> blocks;     // Fock-basis blocks of O
            std::vector<nda::array<dcomplex, 2>> blocks_dag; // Fock-basis blocks of O^dag
        };

        template <bool IsComplex>
        DynintOpBlocks resolve_dynint_op(const triqs_atom_diag_t<IsComplex> &ad, triqs::operators::many_body_operator_complex const &op, int i) {

            int nb = ad.n_subspaces();

            auto context = [&](std::string const &what) {
                std::ostringstream msg;
                msg << "get_operators_and_interactions: dynamical interaction operator " << i << " (" << op << ") " << what;
                return msg.str();
            };

            // get_op_mat throws when two monomials of op take one subspace to different targets.
            auto op_mat_of = [&](triqs::operators::many_body_operator_complex const &o, std::string const &what) {
                try {
                    return triqs_xca::atom_diag::get_op_mat(ad, o);
                } catch (std::exception const &e) {
                    throw std::invalid_argument(context(what + ": atom_diag::get_op_mat failed with \"" + std::string(e.what())
                                                        + "\". Every monomial must take each atom_diag subspace to the same target subspace."));
                }
            };

            auto om   = op_mat_of(op, "is not block resolvable");
            auto conn = nda::vector<int>(nb);
            for (int b = 0; b < nb; ++b) conn(b) = static_cast<int>(om.connection(b));

            // an operator that is zero on every subspace would give a symmetry set of size zero
            bool any_present = false;
            for (int b = 0; b < nb; ++b) any_present = any_present || (conn(b) != -1);
            if (!any_present) throw std::invalid_argument(context("is zero on every atom_diag subspace"));

            // The connection row must be injective for the adjoint to have a well defined connection row. Checked before
            // the adjoint is fetched, whose blocks would otherwise fail as multi-target rather than report the non-injective map.
            std::set<int> seen;
            for (int b = 0; b < nb; ++b) {
                if (conn(b) == -1) continue;
                if (!seen.insert(conn(b)).second) {
                    int b_first = -1;
                    for (int a = 0; a < b; ++a) {
                        if (conn(a) == conn(b)) {
                            b_first = a;
                            break;
                        }
                    }
                    throw std::invalid_argument(context("has a non-injective connection map: subspaces " + std::to_string(b_first) + " and "
                                                        + std::to_string(b) + " are both taken to subspace " + std::to_string(conn(b))
                                                        + ". The block-sparse storage keeps one block per block column and inverts the map to "
                                                          "store the adjoint, so this operator cannot be represented."));
                }
            }

            auto omd      = op_mat_of(triqs::operators::many_body_operator_complex{dagger(op)}, "has an adjoint that is not block resolvable");
            auto conn_dag = nda::vector<int>(nb);
            for (int b = 0; b < nb; ++b) conn_dag(b) = static_cast<int>(omd.connection(b));

            // the connection row of the adjoint is the inverse of the injective row
            for (int b = 0; b < nb; ++b) {
                if (conn(b) != -1 && conn_dag(conn(b)) != b)
                    throw std::invalid_argument(context("has an adjoint whose connection map is not the inverse of its own"));
                if (conn_dag(b) != -1 && conn(conn_dag(b)) != b)
                    throw std::invalid_argument(context("has an adjoint whose connection map is not the inverse of its own"));
            }

            // Rotate the eigenbasis blocks to the Fock basis, as setup_ops_from_triqs_2nd_quant_ops does.
            auto rotate = [&](auto const &m, nda::vector_const_view<int> cn) {
                std::vector<nda::array<dcomplex, 2>> out(nb);
                for (int b = 0; b < nb; ++b) {
                    if (cn(b) == -1) {
                        out[b] = nda::zeros<dcomplex>(1, 1);
                        continue;
                    }
                    auto UR = ad.get_unitary_matrix(b);
                    auto UL = ad.get_unitary_matrix(cn(b));
                    out[b]  = nda::array<dcomplex, 2>{UL * m.block_mat[b] * nda::conj(nda::transpose(UR))};
                }
                return out;
            };

            return {conn, conn_dag, rotate(om, conn), rotate(omd, conn_dag)};
        }

    } // namespace

    template <bool IsComplex>
    std::tuple<BlockOpSymQuartet, nda::vector<long>> get_operators_and_interactions_impl(
        const triqs_atom_diag_t<IsComplex> &ad,
        nda::array_const_view<dcomplex, 3> hyb_coeffs,
        nda::array_const_view<dcomplex, 3> dynint_coeffs,
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {

        int nb    = ad.n_subspaces();
        int p     = hyb_coeffs.extent(0);
        int n_hyb = hyb_coeffs.extent(1);
        int n_int = static_cast<int>(dynint_ops.size());

        if (hyb_coeffs.extent(2) != n_hyb)
            throw std::invalid_argument("get_operators_and_interactions: hyb_coeffs must have shape (p, n_hyb, n_hyb), got ("
                                        + std::to_string(p) + ", " + std::to_string(n_hyb) + ", " + std::to_string(hyb_coeffs.extent(2)) + ")");
        if (n_hyb != static_cast<int>(ad.get_fops().size()))
            throw std::invalid_argument("get_operators_and_interactions: hyb_coeffs covers " + std::to_string(n_hyb)
                                        + " flavours but the atom_diag has " + std::to_string(ad.get_fops().size())
                                        + " fundamental operators");
        if (dynint_coeffs.extent(1) != n_int || dynint_coeffs.extent(2) != n_int)
            throw std::invalid_argument("get_operators_and_interactions: dynint_coeffs must have shape (p, n_int, n_int) with n_int = "
                                        + std::to_string(n_int) + ", got (" + std::to_string(dynint_coeffs.extent(0)) + ", "
                                        + std::to_string(dynint_coeffs.extent(1)) + ", " + std::to_string(dynint_coeffs.extent(2)) + ")");
        if (dynint_coeffs.extent(0) != p)
            throw std::invalid_argument("get_operators_and_interactions: hyb_coeffs and dynint_coeffs must share the pole set, got "
                                        + std::to_string(p) + " and " + std::to_string(dynint_coeffs.extent(0)) + " poles");

        // fermionic symmetry sets, as in get_operators()
        auto sets     = atom_diag::get_operator_sym_sets(ad, n_hyb);
        int n_sym_hyb = static_cast<int>(sets.Fs.size());

        // Resolve every interaction operator before grouping, so that a bad one is reported by index.
        std::vector<DynintOpBlocks> resolved;
        resolved.reserve(n_int);
        for (int i = 0; i < n_int; ++i) resolved.push_back(resolve_dynint_op(ad, dynint_ops[i], i));

        // group the interaction operators by identical connection row, i.e. by sparsity pattern, in order of first appearance
        std::vector<int> group_of_op(n_int, -1);
        std::vector<std::vector<int>> ops_of_group;
        for (int i = 0; i < n_int; ++i) {
            for (int g = 0; g < static_cast<int>(ops_of_group.size()) && group_of_op[i] < 0; ++g) {
                bool same = true;
                for (int b = 0; b < nb && same; ++b) same = (resolved[i].connection(b) == resolved[ops_of_group[g][0]].connection(b));
                if (same) {
                    group_of_op[i] = g;
                    ops_of_group[g].push_back(i);
                }
            }
            if (group_of_op[i] < 0) {
                group_of_op[i] = static_cast<int>(ops_of_group.size());
                ops_of_group.push_back({i});
            }
        }

        // one BlockOpSymSet per group, with the operators in increasing orbital order as assumed by BlockOpSymQuartet
        for (auto const &group : ops_of_group) {
            int qg          = static_cast<int>(group.size());
            auto const &rep = resolved[group[0]];

            std::vector<nda::array<dcomplex, 3>> blocks(nb), blocks_dag(nb);
            for (int b = 0; b < nb; ++b) {
                blocks[b]     = rep.connection(b) == -1 ? nda::zeros<dcomplex>(qg, 1, 1)
                                                        : nda::zeros<dcomplex>(qg, rep.blocks[b].shape(0), rep.blocks[b].shape(1));
                blocks_dag[b] = rep.connection_dag(b) == -1 ? nda::zeros<dcomplex>(qg, 1, 1)
                                                            : nda::zeros<dcomplex>(qg, rep.blocks_dag[b].shape(0), rep.blocks_dag[b].shape(1));
            }
            for (int k = 0; k < qg; ++k) {
                auto const &rk = resolved[group[k]];
                for (int b = 0; b < nb; ++b) {
                    if (rep.connection(b) != -1) blocks[b](k, range::all, range::all) = rk.blocks[b];
                    if (rep.connection_dag(b) != -1) blocks_dag[b](k, range::all, range::all) = rk.blocks_dag[b];
                }
            }
            sets.Fs.emplace_back(rep.connection, blocks);
            sets.F_dags.emplace_back(rep.connection_dag, blocks_dag);
        }

        // concatenate the labels, the interaction sets follow the fermionic ones
        int n_ext   = n_hyb + n_int;
        auto labels = nda::vector<long>(n_ext);
        for (int i = 0; i < n_hyb; ++i) labels(i) = sets.sym_set_labels(i);
        for (int i = 0; i < n_int; ++i) labels(n_hyb + i) = n_sym_hyb + group_of_op[i];

        // the cross-set guard of the quartet constructor validates the extended coefficients uniformly
        auto ext_coeffs = hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs);

        return {BlockOpSymQuartet(sets.Fs, sets.F_dags, ext_coeffs, labels), labels};
    }

    std::tuple<BlockOpSymQuartet, nda::vector<long>> get_operators_and_interactions(
        const triqs_atom_diag_t<true> &ad,
        nda::array_const_view<dcomplex, 3> hyb_coeffs,
        nda::array_const_view<dcomplex, 3> dynint_coeffs,
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {
        return get_operators_and_interactions_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

    std::tuple<BlockOpSymQuartet, nda::vector<long>> get_operators_and_interactions(
        const triqs_atom_diag_t<false> &ad,
        nda::array_const_view<dcomplex, 3> hyb_coeffs,
        nda::array_const_view<dcomplex, 3> dynint_coeffs,
        std::vector<triqs::operators::many_body_operator_complex> const &dynint_ops) {
        return get_operators_and_interactions_impl(ad, hyb_coeffs, dynint_coeffs, dynint_ops);
    }

}
