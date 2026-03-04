
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

using c2py::operator""_a;

// ==================== Wrapped classes =====================

template <>
constexpr bool c2py::is_wrapped<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <>
inline constexpr auto c2py::tp_name<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = "triqs_xca.dlr_dyson_ppsc.DysonItPPSC";
static auto init_0        = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
             cppdlr::dyson_it_ppsc<
                nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
                std::complex<double>>,
             double, cppdlr::imtime_ops,
             const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &>(
      "beta", "itops", "h"),
   c2py::c_constructor<cppdlr::dyson_it_ppsc<nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                                                                     nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
                                                    std::complex<double>>,
                              double, cppdlr::imtime_ops,
                              const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                          nda::borrowed<nda::mem::AddressSpace::Host>> &>("beta", "itops", "g0")};
template <>
constexpr initproc c2py::tp_init<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> =
   init_0.doc(R"DOC(
[1] Constructor for dyson_it

.. note::

   Hamiltonian must either be a symmetric matrix, a Hermitian matrix,
   or a real scalar.

------

[2] Constructor for dyson_it

------

Parameters
----------
beta : {par_0}
   Inverse temperature
itops : {par_1}
   DLR imaginary time object
h : {par_2}
   Hamiltonian
g0 : {par_3}
   Free imaginary time DLR pseudo-particle Green's function
)DOC",
              {{c2py::python_typename<double>()},
               {c2py::python_typename<cppdlr::imtime_ops>()},
               {c2py::python_typename<const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                                                             nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> &>()},
               {c2py::python_typename<const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                  nda::borrowed<nda::mem::AddressSpace::Host>> &>()}});
// solve
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::dyson_it_ppsc<
         nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
         std::complex<double>> &self,
      const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                  nda::borrowed<nda::mem::AddressSpace::Host>> &sig,
      double eta) { return self.solve(sig, eta); },
   "self", "sig", "eta")};

// solve_with_op
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](cppdlr::dyson_it_ppsc<
         nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
         std::complex<double>> &self,
      const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                  nda::borrowed<nda::mem::AddressSpace::Host>> &sig,
      double eta, nda::matrix_view<cppdlr::dcomplex> op) { return self.solve_with_op(sig, eta, op); },
   "self", "sig", "eta", "op")};

static const auto doc_d_0 = fun_0.doc(
   R"DOC(
Solve pseudo-particle Dyson equation for given self-energy

.. note::

   Free Green's function (right hand side of Dyson equation) specified
   at construction of dyson_it object

Parameters
----------
sig : {par_0}
   Self-energy at DLR imaginary time nodes
eta : {par_1}
   Chemical potential

Returns
-------
{ret_0}
   Green's function at DLR imaginary time nodes
)DOC",
   {{c2py::python_typename<const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                       nda::borrowed<nda::mem::AddressSpace::Host>> &>()},
    {c2py::python_typename<double>()}},
   {c2py::python_typename<
      nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});
static const auto doc_d_1 = fun_1.doc(
   R"DOC(
Solve pseudo-particle Dyson equation for given self-energy, chemical potential, and operator.

.. note::

   Free Green's function (right hand side of Dyson equation) specified
   at construction of dyson_it object

Parameters
----------
sig : {par_0}
   Self-energy at DLR imaginary time nodes
eta : {par_1}
   Chemical potential
op : {par_2}
   Static operator (used to vary the chemical potential)

Returns
-------
{ret_0}
   Green's function at DLR imaginary time nodes
)DOC",
   {{c2py::python_typename<const nda::basic_array_view<std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                       nda::borrowed<nda::mem::AddressSpace::Host>> &>()},
    {c2py::python_typename<double>()},
    {c2py::python_typename<nda::matrix_view<cppdlr::dcomplex>>()}},
   {c2py::python_typename<
      nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>>[] = {
   {"solve", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
   {"solve_with_op", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>>[] = {

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<cppdlr::dyson_it_ppsc<
   nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
   std::complex<double>>> = R"DOC(Class for solving the pseudo-particle Dyson equation in imaginary time)DOC" + std::string{"\n\n----------\n\n"}
   + c2py::tp_ctor_doc<cppdlr::dyson_it_ppsc<
      nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
      std::complex<double>>>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "dlr_dyson_ppsc",  /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_dlr_dyson_ppsc() {

  if (not c2py::check_python_version("dlr_dyson_ppsc")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(
         &c2py::wrap_pytype<cppdlr::dyson_it_ppsc<
            nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
            std::complex<double>>>)
      < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<cppdlr::dyson_it_ppsc<
     nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,
     std::complex<double>>>("DysonItPPSC", m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
