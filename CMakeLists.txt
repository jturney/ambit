#
#  Copyright (C) 2014 Justin Turney
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

cmake_policy(SET CMP0048 NEW)  # project_VERSION* variables populated from project(... VERSION x.x.x) string
cmake_policy(SET CMP0079 NEW)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
    cmake_policy(SET CMP0135 NEW)
endif()

cmake_minimum_required(VERSION 3.16)

project(ambit
        VERSION 0.6.0  # SYNC WITH `project(pyambit VERSION <>)` in pyambit/CMakeLists.txt
        LANGUAGES C CXX)
set(ambit_AUTHORS      "Justin M. Turney")
set(ambit_DESCRIPTION  "C++ library for the implementation of tensor product calculations")
set(ambit_URL          "https://github.com/jturney/ambit")
set(ambit_LICENSE      "GNU Lesser General Public License v3 (LGPL-3.0)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TargetOpenMP_FIND_COMPONENTS "CXX")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# <<<  CMake includes  >>>

include (CheckCXXCompilerFlag)
include (CMakePackageConfigHelpers)

# <<<  Default CMake options  >>>
include(psi4OptionsTools)

option_with_default(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Release)
#option_with_print(BUILD_SHARED_LIBS "Build final library as shared, not static" OFF)
#option_with_default(BUILD_FPIC "Libraries will be compiled with position independent code" ON)
#if(${BUILD_SHARED_LIBS} AND NOT ${BUILD_FPIC})
#    message(FATAL_ERROR "BUILD_SHARED_LIBS ON and BUILD_FPIC OFF are incompatible, as shared library requires position independent code")
#endif()

option (BUILD_DOXYGEN        "Use Doxygen to create a HTML/PDF manual" OFF)
option (BUILD_SPHINX         "Build the user manual with Sphinx"       OFF)
option (STATIC_ONLY          "Compile only the static library"         OFF)
option (SHARED_ONLY          "Compile only the shared library"         OFF)
option (ENABLE_TESTS         "Compile the tests"                       ON)
option (WITH_MPI             "Build the library with MPI"              OFF)
option (ENABLE_CYCLOPS       "Enable Cyclops usage" OFF)
option (BUILD_FPIC           "Static library in STATIC_ONLY will be compiled with position independent code" ON)
option (CYCLOPS              "Location of the Cyclops build directory" "")
option (ELEMENTAL            "Location of the Elemental build directory" "")
option_with_print(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
include(xhost)  # defines: option(ENABLE_XHOST "Enable processor-specific optimization" ON)
option_with_default(FC_SYMBOL "The type of Fortran name mangling" 2)
option_with_default(BUILD_FPIC "Compile static libraries with position independent code" ON)
option_with_default(CMAKE_INSTALL_LIBDIR "Directory to which libraries installed" lib)
option_with_default(PYMOD_INSTALL_LIBDIR "If ambit_ENABLE_PYTHON, location within CMAKE_INSTALL_LIBDIR to which python modules are installed" /)
option_with_default(ENABLE_GENERIC "Enables mostly static linking of system and math libraries for shared library" OFF)
option_with_default(ambit_INSTALL_CMAKEDIR "Directory to which ambit CMake config files installed." share/cmake/ambit)
option_with_print(ambit_ENABLE_PYTHON "Build Python bindings (requires Python and Pybind11)" ON)

########################  Process & Validate Options  ##########################
include(GNUInstallDirs)
include(autocmake_safeguards)
#include(custom_color_messages)
# external projects manage their own OpenMP and c++YY flags, so only add to CXX_FLAGS for psi4-core
include(autocmake_omp)
include(custom_cxxstandard)
include(custom_static_library)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local/ambit" CACHE PATH "Install path" FORCE)
endif()
message(STATUS "Ambit install path: ${CMAKE_INSTALL_PREFIX}")

if (STATIC_ONLY AND SHARED_ONLY)
    message (FATAL_ERROR "The options STATIC_ONLY=ON and SHARED_ONLY=ON are conflicting." )
endif()

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries("${CMAKE_THREAD_LIBS_INIT}")
#link_libraries(Threads::Threads)

# HDF5 detection

find_package(TargetHDF5 REQUIRED)
include_directories(SYSTEM $<TARGET_PROPERTY:tgt::hdf5,INTERFACE_INCLUDE_DIRECTORIES>)

# BLAS and LAPACK
find_package (TargetLAPACK REQUIRED)


#if (NOT ENABLE_STATIC AND NOT ENABLE_PSI4)
#    include(ConfigPython)
#    link_libraries("${Python_LIBRARIES}")
#    include_directories(SYSTEM "${Python_INCLUDE_DIRS}")
#endif()

#if (ENABLE_PSI4)
#    include_directories(SYSTEM ${PSI4_INCLUDE_DIRS})
#endif()

# Append the suffix given from input to all generated executables
#if(EXECUTABLE_SUFFIX)
#   set(CMAKE_EXECUTABLE_SUFFIX "${EXECUTABLE_SUFFIX}")
#   message(STATUS "Suffix ${CMAKE_EXECUTABLE_SUFFIX} will be appended to executables")
#endif()
#
#if(LDFLAGS)
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LDFLAGS}")
#endif()
#
# tensor-specific includes and libraries
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
)

if(ENABLE_CYCLOPS AND CYCLOPS)
    include_directories(${CYCLOPS}/include)
    add_definitions(-DHAVE_CYCLOPS)
endif()
if (ENABLE_ELEMENTAL AND ELEMENTAL)
    include_directories(${ELEMENTAL}/include)
    add_definitions(-DHAVE_ELEMENTAL)
endif()

#if (ENABLE_PSI4 AND PSI4_SOURCE_DIR AND PSI4_BINARY_DIR)
#    add_definitions(-DENABLE_PSI4=1)
#    include_directories(
#            ${PSI4_SOURCE_DIR}/include
#            ${PSI4_SOURCE_DIR}/src/lib
#            ${PSI4_BINARY_DIR}/include
#            ${PSI4_BINARY_DIR}/src/lib
#    )
#endif()

set(CMAKECONFIG_INSTALL_DIR "${ambit_INSTALL_CMAKEDIR}")

# include directory (for installation)
add_subdirectory(include)

# recursively add source directories
add_subdirectory(src)

if (ENABLE_TESTS)
    # test suite
    enable_testing()
    include(CTest)
    add_subdirectory(test)
endif()

# sample suite
add_subdirectory(samples)

# Python directory
if (ambit_ENABLE_PYTHON)
    add_subdirectory(pyambit)
endif()

# Add all targets to the build-tree export set

# <<<  Export Config  >>>

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/${PROJECT_NAME}Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${${PROJECT_NAME}_VERSION}
        COMPATIBILITY SameMajorVersion)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    cmake/FindTargetLAPACK.cmake
    cmake/FindTargetHDF5.cmake
  DESTINATION
    ${CMAKECONFIG_INSTALL_DIR}
  )

# this has to be the very last CMake module to be included
#include(ConfigInfo)
