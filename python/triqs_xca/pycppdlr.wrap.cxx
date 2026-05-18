
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

// ==================== enums =====================

template <>
std::map<cppdlr::statistic_t, str_t> c2py::enum_to_string<cppdlr::statistic_t> = {{cppdlr::statistic_t::Boson, "Boson"},
                                                                                  {cppdlr::statistic_t::Fermion, "Fermion"}};

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0                                            = cppdlr::imtime_ops;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca.pycppdlr.ImTimeOps";
static auto _c2py_init_0                                     = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                          _c2py_cls_0, double,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>, bool>(
      "lambda", "dlr_rf", "symmetrize"),
   c2py::c_constructor<
                                          _c2py_cls_0, double,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>(
      "lambda", "dlr_rf"),
   c2py::c_constructor<
                                          _c2py_cls_0, double,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const double, 2, nda::C_stride_layout, 'M', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const double, 2, nda::C_stride_layout, 'M', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const int, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>(
      "lambda", "dlr_rf", "dlr_it", "cf2it", "it2cf_lu", "it2cf_piv"),
   c2py::c_constructor<_c2py_cls_0>()};
template <> constexpr initproc c2py::tp_init<_c2py_cls_0>    = c2py::pyfkw_constructor<_c2py_init_0>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(R"DOC()DOC");
// build_evalvec
static auto const _c2py_fun_0 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self, double t) -> decltype(auto) { return self.build_evalvec(t); }, "self", "t")};

// coefs2eval
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc,
      double t) -> decltype(auto) {
     return self.template coefs2eval<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<>>>,
                                     std::complex<double>>(gc, t);
   },
   "self", "gc", "t")};

// coefs2vals
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc)
      -> decltype(auto) {
     return self.template coefs2vals<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<>>>,
                                     std::complex<double>>(gc);
   },
   "self", "gc")};

// convolve
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 const &self, double beta,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &fc,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &gc,
      bool time_order) -> decltype(auto) {
     return self.template convolve<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<>>>,
                                   std::complex<double>>(beta, fc, gc, time_order);
   },
   "self", "beta", "fc", "gc", "time_order"_a = false)};

// convolve_init
static auto const _c2py_fun_4 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.convolve_init(); }, "self")};

// get_cf2it
static auto const _c2py_fun_5 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_cf2it(); }, "self")};

// get_ipmat
static auto const _c2py_fun_6 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_ipmat(); }, "self")};

// get_it2cf_lu
static auto const _c2py_fun_7 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_it2cf_lu(); }, "self")};

// get_it2cf_piv
static auto const _c2py_fun_8 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_it2cf_piv(); }, "self")};

// get_it2cf_zlu
static auto const _c2py_fun_9 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_it2cf_zlu(); }, "self")};

// get_itnodes
static auto const _c2py_fun_10 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_itnodes(); }, "self"),
                           c2py::cmethod([](_c2py_cls_0 const &self, int i) -> decltype(auto) { return self.get_itnodes(i); }, "self", "i")};

// get_rfnodes
static auto const _c2py_fun_11 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.get_rfnodes(); }, "self"),
                           c2py::cmethod([](_c2py_cls_0 const &self, int i) -> decltype(auto) { return self.get_rfnodes(i); }, "self", "i")};

// innerprod_init
static auto const _c2py_fun_12 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.innerprod_init(); }, "self")};

// lambda
static auto const _c2py_fun_13 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.lambda(); }, "self")};

// rank
static auto const _c2py_fun_14 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.rank(); }, "self")};

// reflect
static auto const _c2py_fun_15 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &g)
      -> decltype(auto) {
     return self.template reflect<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<>>>>(g);
   },
   "self", "g")};

// reflect_init
static auto const _c2py_fun_16 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.reflect_init(); }, "self")};

// tconvolve_init
static auto const _c2py_fun_17 =
   c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 const &self) -> decltype(auto) { return self.tconvolve_init(); }, "self")};

// vals2coefs
static auto const _c2py_fun_18 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 const &self,
      const nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &g,
      bool transpose) -> decltype(auto) {
     return self.template vals2coefs<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<>>>,
                                     std::complex<double>>(g, transpose);
   },
   "self", "g", "transpose"_a = false)};

static const auto _c2py_doc_0  = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1  = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2  = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3  = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4  = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5  = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6  = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7  = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8  = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9  = _c2py_fun_9.doc(R"DOC()DOC");
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");
static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");
static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
static const auto _c2py_doc_15 = _c2py_fun_15.doc(R"DOC()DOC");
static const auto _c2py_doc_16 = _c2py_fun_16.doc(R"DOC()DOC");
static const auto _c2py_doc_17 = _c2py_fun_17.doc(R"DOC()DOC");
static const auto _c2py_doc_18 = _c2py_fun_18.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"build_evalvec", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"coefs2eval", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"coefs2vals", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"convolve", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"convolve_init", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"get_cf2it", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"get_ipmat", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {"get_it2cf_lu", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {"get_it2cf_piv", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {"get_it2cf_zlu", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
   {"get_itnodes", (PyCFunction)c2py::pyfkw<_c2py_fun_10>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
   {"get_rfnodes", (PyCFunction)c2py::pyfkw<_c2py_fun_11>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
   {"innerprod_init", (PyCFunction)c2py::pyfkw<_c2py_fun_12>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_12.c_str()},
   {"lambda", (PyCFunction)c2py::pyfkw<_c2py_fun_13>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
   {"rank", (PyCFunction)c2py::pyfkw<_c2py_fun_14>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
   {"reflect", (PyCFunction)c2py::pyfkw<_c2py_fun_15>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_15.c_str()},
   {"reflect_init", (PyCFunction)c2py::pyfkw<_c2py_fun_16>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_16.c_str()},
   {"tconvolve_init", (PyCFunction)c2py::pyfkw<_c2py_fun_17>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_17.c_str()},
   {"vals2coefs", (PyCFunction)c2py::pyfkw<_c2py_fun_18>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_18.c_str()},
   {"__write_hdf5__", c2py::tpxx_write_h5<_c2py_cls_0>, METH_VARARGS, "  "},
   {"__getstate__", c2py::getstate_tuple<_c2py_cls_0>, METH_NOARGS, ""},
   {"__setstate__", c2py::setstate_tuple<_c2py_cls_0>, METH_O, ""},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <> const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

// build_dlr_rf
static auto const _c2py_fun_19 =
   c2py::dispatcher_f_kw_t{c2py::cfun([](double lambda, double eps, bool symmetrize) { return cppdlr::build_dlr_rf(lambda, eps, symmetrize); },
                                      "lambda", "eps", "symmetrize"),
                           c2py::cfun([](double lambda, double eps) { return cppdlr::build_dlr_rf(lambda, eps); }, "lambda", "eps")};

static const auto _c2py_doc_19 = _c2py_fun_19.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"build_dlr_rf", (PyCFunction)c2py::pyfkw<_c2py_fun_19>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_19.c_str()},
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
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "ImTimeOps");
#undef _add_type

  c2py::pyref module = c2py::pyref::module("h5.formats");
  if (not module) return nullptr;
  c2py::pyref register_class = module.attr("register_class");

  register_h5_type<_c2py_cls_0>(register_class);

  return m;
}
#endif
// CLAIR_WRAP_GEN
