cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME scifem)
# nanobind uses aligned deallocators only present on macOS > 10.14
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(${PROJECT_NAME} VERSION "0.1.0.0" LANGUAGES C CXX)

if (WIN32)
    # Windows requires all symbols to be manually exported.
    # This flag exports all symbols automatically, as in Unix.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import os, sys, basix; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_VARIABLE BASIX_ERROR_OUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

if(Basix_FOUND)
  message(STATUS "Found Basix at ${Basix_DIR}")
endif()



find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  _scifem
  STABLE_ABI
  src/scifem.cpp
)
target_link_libraries(_scifem PRIVATE dolfinx)

set_target_properties(_scifem PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install directive for scikit-build-core
install(TARGETS _scifem LIBRARY DESTINATION ${PROJECT_NAME})