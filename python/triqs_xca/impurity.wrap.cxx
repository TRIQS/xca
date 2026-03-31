
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

// ==================== enums =====================



// ==================== module classes =====================


// --------- class _c2py_cls_0 -----------
using _c2py_cls_0 = fastdiagram;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0> = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca.impurity.Fastdiagram"; static auto _c2py_init_0 = c2py::dispatcher_c_kw_t { 
 c2py::c_constructor<_c2py_cls_0,double,double,cppdlr::imtime_ops,nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>,nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>( "beta", "lambda", "itops", "F", "F_dag")};
 template <> constexpr initproc c2py::tp_init<_c2py_cls_0> = c2py::pyfkw_constructor<_c2py_init_0>;
template <> const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(R"DOC(
Constructor for fastdiagram, construct itops and diagram topology matrices

Parameters
----------
beta : {par_0}
   inverse temperature
lambda : {par_1}
   DLR cutoff parameter
itops : {par_2}
   CPPDLR imaginary time operations object
F : {par_3}
   impurity annihilation operator in pseudo-particle space, of size n*N*N
F_dag : {par_4}
   impurity creation operator in pseudo-particle space, of size n*N*N
)DOC", {{c2py::python_typename<double>()}, {c2py::python_typename<double>()}, {c2py::python_typename<cppdlr::imtime_ops>()}, {c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()}, {c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()}});
 // G_calc
                             static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Gt,std::string order) -> decltype(auto) { return self.G_calc(Gt,order); }, "self" ,  "Gt", "order") };

 // G_calc_group
                             static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Gt,nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> D,nda::basic_array<int, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> diagramindex) -> decltype(auto) { return self.G_calc_group(Gt,D,diagramindex); }, "self" ,  "Gt", "D", "diagramindex") };

 // Sigma_calc
                             static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Gt,std::string order) -> decltype(auto) { return self.Sigma_calc(Gt,order); }, "self" ,  "Gt", "order") };

 // Sigma_calc_group
                             static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Gt,nda::basic_array<int, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> D,nda::basic_array<int, 1, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> diagramindex) -> decltype(auto) { return self.Sigma_calc_group(Gt,D,diagramindex); }, "self" ,  "Gt", "D", "diagramindex") };

 // copy_aaa_result
                             static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<double, 1, nda::C_layout, 'V', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> pol0,nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> weights0) -> decltype(auto) { return self.copy_aaa_result(pol0,weights0); }, "self" ,  "pol0", "weights0") };

 // free_greens
                             static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , double beta,nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> H_S,double mu,bool time_order) -> decltype(auto) { return self.free_greens(beta,H_S,mu,time_order); }, "self" ,  "beta", "H_S", "mu"_a = 0.0, "time_order"_a = false) };

 // free_greens_ppsc
                             static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , double beta,nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> H_S) -> decltype(auto) { return self.free_greens_ppsc(beta,H_S); }, "self" ,  "beta", "H_S") };

 // get_it_actual
                             static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self   ) -> decltype(auto) { return self.get_it_actual(); }, "self"   ) };

 // hyb_decomposition
                             static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , bool poledlrflag,double eps) -> decltype(auto) { return self.hyb_decomposition(poledlrflag,eps); }, "self" ,  "poledlrflag"_a = true, "eps"_a = 0.0) };

 // hyb_init
                             static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Deltat0,bool poledlrflag) -> decltype(auto) { return self.hyb_init(Deltat0,poledlrflag); }, "self" ,  "Deltat0", "poledlrflag"_a = true) };

 // number_of_diagrams
                             static auto const _c2py_fun_10 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , int m) -> decltype(auto) { return self.number_of_diagrams(m); }, "self" ,  "m") };

 // partition_function
                             static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> Gt) -> decltype(auto) { return self.partition_function(Gt); }, "self" ,  "Gt") };

 // time_ordered_dyson
                             static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{  c2py::cmethod([](_c2py_cls_0  & self , double beta,nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>> H_S,double eta_0,nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> Sigma_t) -> decltype(auto) { return self.time_ordered_dyson(beta,H_S,eta_0,Sigma_t); }, "self" ,  "beta", "H_S", "eta_0", "Sigma_t") };

 static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC(
Compute impurity Green's function diagram of certain order, given pseudo-particle Green's function G(t)

Parameters
----------
Gt : {par_0}
   pseudo-particle Green's function G(t), of size r*N*N
order : {par_1}
   diagram order: "NCA", "OCA" or "TCA"

Returns
-------
{ret_0}
   impurity Green's function diagram, r*n*n
)DOC", {{c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()}, {c2py::python_typename<std::string>()}}, {c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});
 static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
 static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC(
Compute pseudo-particle self energy diagram of certain order, given pseudo-particle Green's function G(t)

Parameters
----------
Gt : {par_0}
   pseudo-particle Green's function G(t), of size r*N*N
order : {par_1}
   diagram order: "NCA", "OCA" or "TCA"

Returns
-------
{ret_0}
   pseudo-particle self energy diagram, r*N*N
)DOC", {{c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()}, {c2py::python_typename<std::string>()}}, {c2py::python_typename<nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});
 static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
 static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
 static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC(
free green's function, wrapped from free_gf of cppdlr
)DOC");
 static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC(
free pseudo-particle green's function, wrapped from free_gf_ppsc
)DOC");
 static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
 static const auto _c2py_doc_8 = _c2py_fun_8.doc(R"DOC(
calculate decomposition and reflection of hybridization Deltat

Parameters
----------
poledlrflag : {par_0}
   flag for whether to use dlr for pole expansion. True for using dlr. False has not been implemented yet.
eps : {par_1}
   SVD truncation threshold
)DOC", {{c2py::python_typename<bool>()}, {c2py::python_typename<double>()}});
 static const auto _c2py_doc_9 = _c2py_fun_9.doc(R"DOC()DOC");
 static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");
 static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
 static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");

      // ----- Method table ----
      template <> PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
            {"G_calc", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_0.c_str()},  {"G_calc_group", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_1.c_str()},  {"Sigma_calc", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_2.c_str()},  {"Sigma_calc_group", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_3.c_str()},  {"copy_aaa_result", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_4.c_str()},  {"free_greens", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_5.c_str()},  {"free_greens_ppsc", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_6.c_str()},  {"get_it_actual", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_7.c_str()},  {"hyb_decomposition", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_8.c_str()},  {"hyb_init", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_9.c_str()},  {"number_of_diagrams", (PyCFunction)c2py::pyfkw<_c2py_fun_10>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_10.c_str()},  {"partition_function", (PyCFunction)c2py::pyfkw<_c2py_fun_11>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_11.c_str()},  {"time_ordered_dyson", (PyCFunction)c2py::pyfkw<_c2py_fun_12>, METH_VARARGS | METH_KEYWORDS , _c2py_doc_12.c_str()}, 
           {nullptr, nullptr, 0, nullptr} // Sentinel
      };

      constexpr auto _c2py_doc_member_0 = R"DOC()DOC";  constexpr auto _c2py_doc_member_1 = R"DOC()DOC";  constexpr auto _c2py_doc_member_2 = R"DOC()DOC"; 

      // ----- Member and property table ----

      template <> constinit PyGetSetDef c2py::tp_getset<_c2py_cls_0>[] = {
          c2py::getsetdef_from_member<&_c2py_cls_0::Deltaiw, _c2py_cls_0>("Deltaiw", _c2py_doc_member_0), c2py::getsetdef_from_member<&_c2py_cls_0::Deltaiw_reflect, _c2py_cls_0>("Deltaiw_reflect", _c2py_doc_member_1), c2py::getsetdef_from_member<&_c2py_cls_0::dlr_if_dense, _c2py_cls_0>("dlr_if_dense", _c2py_doc_member_2),
         
         {nullptr,nullptr,nullptr,nullptr,nullptr }
      };

      
template <> const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC(Class responsible for fast diagram calculation of a given impurity problem using hybridization expansion.)DOC" +  std::string{"\n\n----------\n\n"}  + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================




//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
 {nullptr, nullptr, 0, nullptr}  // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT, "impurity", /* name of module */
   R"RAWDOC()RAWDOC",                        /* module documentation, may be NULL */
   -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
   module_methods, NULL, NULL, NULL, NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_impurity() {

  if (not c2py::check_python_version("impurity")) return NULL;

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
  _add_type(_c2py_cls_0, "Fastdiagram"); 
#undef _add_type

  
  
  
  

  return m;
}
#endif
// CLAIR_WRAP_GEN
