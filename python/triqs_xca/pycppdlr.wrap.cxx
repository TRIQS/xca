
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>
#include <c2py/serialization/h5.hpp>

using c2py::operator""_a;

// ==================== Wrapped classes =====================

template <> constexpr bool c2py::is_wrapped<cppdlr::imtime_ops> = true;

// ==================== enums =====================

template <>
std::map<cppdlr::statistic_t, str_t> c2py::enum_to_string<cppdlr::statistic_t> = {{cppdlr::statistic_t::Boson, "Boson"},
                                                                                  {cppdlr::statistic_t::Fermion, "Fermion"}};

// ==================== module classes =====================

template <> inline constexpr auto c2py::tp_name<cppdlr::imtime_ops> = "triqs_xca.pycppdlr.ImTimeOps";
static auto init_0                                                  = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<cppdlr::imtime_ops, double, nda::vector_const_view<double>, bool>("lambda", "dlr_rf", "symmetrize"),
   c2py::c_constructor<cppdlr::imtime_ops, double, nda::vector_const_view<double>>("lambda", "dlr_rf"),
   c2py::c_constructor<cppdlr::imtime_ops, double, nda::vector_const_view<double>, nda::vector_const_view<double>, nda::matrix_const_view<double>,
                                                                        nda::matrix_const_view<double>, nda::vector_const_view<int>>("lambda", "dlr_rf", "dlr_it", "cf2it", "it2cf_lu", "it2cf_piv"),
   c2py::c_constructor<cppdlr::imtime_ops>()};
template <> constexpr initproc c2py::tp_init<cppdlr::imtime_ops>    = c2py::pyfkw_constructor<init_0>;
template <> const std::string c2py::tp_ctor_doc<cppdlr::imtime_ops> = init_0.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
// build_evalvec
static auto const fun_0 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self, double t) { return self.build_evalvec(t); }, "self", "t")};

// coefs2eval
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::imtime_ops const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc,
      double t) { return self.coefs2eval(gc, t); },
   "self", "gc", "t")};

// coefs2vals
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::imtime_ops const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc) {
     return self.coefs2vals(gc);
   },
   "self", "gc")};

// convolve
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::imtime_ops const &self, double beta,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &fc,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc,
      bool time_order) { return self.convolve(beta, fc, gc, time_order); },
   "self", "beta", "fc", "gc", "time_order"_a = false)};

// convolve_init
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.convolve_init(); }, "self")};

// get_cf2it
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_cf2it(); }, "self")};

// get_ipmat
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_ipmat(); }, "self")};

// get_it2cf_lu
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_it2cf_lu(); }, "self")};

// get_it2cf_piv
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_it2cf_piv(); }, "self")};

// get_it2cf_zlu
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_it2cf_zlu(); }, "self")};

// get_itnodes
static auto const fun_10 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_itnodes(); }, "self"),
                           c2py::cmethod([](cppdlr::imtime_ops const &self, int i) { return self.get_itnodes(i); }, "self", "i")};

// get_rfnodes
static auto const fun_11 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.get_rfnodes(); }, "self"),
                           c2py::cmethod([](cppdlr::imtime_ops const &self, int i) { return self.get_rfnodes(i); }, "self", "i")};

// hdf5_format
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cfun([]() { return cppdlr::imtime_ops::hdf5_format(); })};

// innerprod_init
static auto const fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.innerprod_init(); }, "self")};

// lambda
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.lambda(); }, "self")};

// rank
static auto const fun_15 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.rank(); }, "self")};

// reflect
static auto const fun_16 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::imtime_ops const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &g) {
     return self.reflect(g);
   },
   "self", "g")};

// reflect_init
static auto const fun_17 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.reflect_init(); }, "self")};

// tconvolve_init
static auto const fun_18 = c2py::dispatcher_f_kw_t{c2py::cmethod([](cppdlr::imtime_ops const &self) { return self.tconvolve_init(); }, "self")};

// vals2coefs
static auto const fun_19 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::imtime_ops const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &g,
      bool transpose) { return self.vals2coefs(g, transpose); },
   "self", "g", "transpose"_a = false)};

static const auto doc_d_0  = fun_0.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_1  = fun_1.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_2  = fun_2.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_3  = fun_3.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_4  = fun_4.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_5  = fun_5.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_6  = fun_6.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_7  = fun_7.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_8  = fun_8.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_9  = fun_9.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_10 = fun_10.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_11 = fun_11.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_12 = fun_12.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_14 = fun_14.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_15 = fun_15.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_16 = fun_16.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_17 = fun_17.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_18 = fun_18.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
static const auto doc_d_19 = fun_19.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<cppdlr::imtime_ops>[] = {
   {"build_evalvec", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
   {"coefs2eval", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
   {"coefs2vals", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
   {"convolve", (PyCFunction)c2py::pyfkw<fun_3>, METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
   {"convolve_init", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
   {"get_cf2it", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS, doc_d_5.c_str()},
   {"get_ipmat", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS, doc_d_6.c_str()},
   {"get_it2cf_lu", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS, doc_d_7.c_str()},
   {"get_it2cf_piv", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS, doc_d_8.c_str()},
   {"get_it2cf_zlu", (PyCFunction)c2py::pyfkw<fun_9>, METH_VARARGS | METH_KEYWORDS, doc_d_9.c_str()},
   {"get_itnodes", (PyCFunction)c2py::pyfkw<fun_10>, METH_VARARGS | METH_KEYWORDS, doc_d_10.c_str()},
   {"get_rfnodes", (PyCFunction)c2py::pyfkw<fun_11>, METH_VARARGS | METH_KEYWORDS, doc_d_11.c_str()},
   {"hdf5_format", (PyCFunction)c2py::pyfkw<fun_12>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, doc_d_12.c_str()},
   {"innerprod_init", (PyCFunction)c2py::pyfkw<fun_13>, METH_VARARGS | METH_KEYWORDS, doc_d_13.c_str()},
   {"lambda", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS, doc_d_14.c_str()},
   {"rank", (PyCFunction)c2py::pyfkw<fun_15>, METH_VARARGS | METH_KEYWORDS, doc_d_15.c_str()},
   {"reflect", (PyCFunction)c2py::pyfkw<fun_16>, METH_VARARGS | METH_KEYWORDS, doc_d_16.c_str()},
   {"reflect_init", (PyCFunction)c2py::pyfkw<fun_17>, METH_VARARGS | METH_KEYWORDS, doc_d_17.c_str()},
   {"tconvolve_init", (PyCFunction)c2py::pyfkw<fun_18>, METH_VARARGS | METH_KEYWORDS, doc_d_18.c_str()},
   {"vals2coefs", (PyCFunction)c2py::pyfkw<fun_19>, METH_VARARGS | METH_KEYWORDS, doc_d_19.c_str()},
   {"__write_hdf5__", c2py::tpxx_write_h5<cppdlr::imtime_ops>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_tuple<cppdlr::imtime_ops>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_tuple<cppdlr::imtime_ops>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<cppdlr::imtime_ops>[] = {

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <> const std::string c2py::tp_doc<cppdlr::imtime_ops> = R"DOC()DOC" + c2py::tp_ctor_doc<cppdlr::imtime_ops>;

// ==================== module functions ====================

// build_dlr_rf
static auto const fun_20 =
   c2py::dispatcher_f_kw_t{c2py::cfun([](double lambda, double eps, bool symmetrize) { return cppdlr::build_dlr_rf(lambda, eps, symmetrize); },
                                      "lambda", "eps", "symmetrize"),
                           c2py::cfun([](double lambda, double eps) { return cppdlr::build_dlr_rf(lambda, eps); }, "lambda", "eps")};

static const auto doc_d_20 = fun_20.doc(R"DOC()DOC", std::vector<std::string>{}, std::vector<std::string>{});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"build_dlr_rf", (PyCFunction)c2py::pyfkw<fun_20>, METH_VARARGS | METH_KEYWORDS, doc_d_20.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "pycppdlr",        /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_pycppdlr() {

  if (not c2py::check_python_version("pycppdlr")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<cppdlr::imtime_ops>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<cppdlr::imtime_ops>("ImTimeOps", m, conv_table);

  c2py::pyref module = c2py::pyref::module("h5.formats");
  if (not module) return nullptr;
  c2py::pyref register_class = module.attr("register_class");

  register_h5_type<cppdlr::imtime_ops>(register_class);

  return m;
}
#endif
// CLAIR_WRAP_GEN
