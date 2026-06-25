# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  pycppdlr
  load_module
  spinless_dimer
  spinless_dimer_solver
  offdiagonal_nca
  test_diag
  unitary_invariance
  h5_io
  h5_io_bs
  dense
  triqs_solver
  one_spinful_fermion
  analytic_diagrams
  block_sparse_copy
  block_sparse_api
  block_sparse_api_one_fermion
  block_sparse_api_two_fermions
  block_sparse_self_cons
  block_sparse_bethe
  block_sparse_normalization
  bug_bs_one_fermion
  volterra_integral
  one_time_correlator
  expectation_value
  )

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
    "PYTHONPATH=path_list_prepend:${pyed_SOURCE_DIR}"
    "PYTHONPATH=path_list_prepend:${adapol_SOURCE_DIR}/src"
    "PYTHONPATH=path_list_prepend:${PROJECT_BINARY_DIR}/python")
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT ${SANITIZER_RT_PRELOAD})
endforeach()
