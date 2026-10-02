# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
  else()
    message(STATUS "cppcheck not found in $PATH. Please consider installing cppcheck for additional checks!")
  endif()

endif()

# ========= Dynamic Analyzer Checks ==========

option(ASAN OFF "Compile library and executables with LLVM Address Sanitizer")
if(ASAN AND NOT TARGET asan)
  find_package(sanitizer REQUIRED COMPONENTS asan)
endif()

option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")
if(UBSAN AND NOT TARGET ubsan)
  find_package(sanitizer REQUIRED COMPONENTS ubsan)
endif()

option(MSAN OFF "Compile library and executables with LLVM Memory Sanitizer")
if(MSAN AND NOT TARGET msan)
  find_package(sanitizer REQUIRED COMPONENTS msan)
endif()

option(TSAN OFF "Compile library and executables with LLVM Thread Sanitizer")
if(TSAN AND NOT TARGET tsan)
  find_package(sanitizer REQUIRED COMPONENTS tsan)
endif()

# ========= C++ libraries ==========

# Build and install the static library <unit>_c from all sources in the current directory,
# with headers installed as include/<unit>/*.hpp. Used by the units below.
function(xca_add_library unit)
  file(GLOB_RECURSE sources *.cpp)
  add_library(${unit}_c ${sources})
  add_library(${PROJECT_NAME}::${unit}_c ALIAS ${unit}_c)

  # Link against triqs and the cppdlr overlay (which carries the common compile definitions), and enable warnings
  target_link_libraries(${unit}_c PUBLIC ${PROJECT_NAME}_cppdlr PRIVATE $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)

  # Configure target and compilation
  set_target_properties(${unit}_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  )
  target_include_directories(${unit}_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
  target_include_directories(${unit}_c SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
  target_link_directories(${unit}_c PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_link_directories(${unit}_c INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_LIBDIR}>)
  # Install library and headers
  install(TARGETS ${unit}_c EXPORT ${PROJECT_NAME}-targets DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

  # Static analyzers
  if(ANALYZE_SOURCES)
    if(CLANG_TIDY_EXECUTABLE)
      set_target_properties(${unit}_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
    endif()
    if(CPPCHECK_EXECUTABLE)
      add_custom_command(
        TARGET ${unit}_c
        COMMAND ${CPPCHECK_EXECUTABLE}
          --enable=warning,style,performance,portability
          --std=c++23
          --template=gcc
          --verbose
          --force
          --quiet
          ${sources}
        WORKING_DIRECTORY
          ${CMAKE_CURRENT_SOURCE_DIR}
      )
    endif()
  endif()

  # Sanitizers
  foreach(sanitizer asan ubsan msan tsan)
    string(TOUPPER ${sanitizer} SANITIZER)
    if(${SANITIZER})
      target_link_libraries(${unit}_c PUBLIC $<BUILD_INTERFACE:${sanitizer}>)
    endif()
  endforeach()
endfunction()

# cppdlr overlay: the header-only additions to the external cppdlr library
add_subdirectory(cppdlr)

# triqs_soehyb: the legacy dense hybridization expansion solver
add_subdirectory(triqs_soehyb)

# triqs_xca: the block-sparse arbitrary crossing approximation solver
add_subdirectory(triqs_xca)
