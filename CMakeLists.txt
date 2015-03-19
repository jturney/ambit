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

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ambit C CXX)

set(AMBIT_MAJOR_VERSION 0)
set(AMBIT_MINOR_VERSION 1)
set(AMBIT_PATCH_VERSION 0)
set(AMBIT_VERSION ${AMBIT_MAJOR_VERSION}.${AMBIT_MINOR_VERSION}.${AMBIT_PATCH_VERSION})

# Custom CMake modules location
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_SOURCE_DIR}/cmake/compilers
    ${CMAKE_SOURCE_DIR}/cmake/math
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Options
option(ENABLE_MPI           "Enable MPI parallelization"  OFF)
option(ENABLE_CYCLOPS       "Enable Cyclops usage" OFF)
option(ENABLE_OMP           "Enable OpenMP parallelization" ON)
option(ENABLE_AUTO_BLAS     "Enable CMake to autodetect BLAS" ON)
option(ENABLE_AUTO_LAPACK   "Enable CMake to autodetect LAPACK" ON)
option(ENABLE_ACCELERATE    "Enable use of Mac OS X Accelerate Framework" OFF)
option(ENABLE_PYTHON        "Enable Python interface" ON)
option(ENABLE_CXX11_SUPPORT "Enable C++11 compiler support" ON)

option(CYCLOPS              "Location of the Cyclops build directory" "")
option(ELEMENTAL            "Location of the Elemental build directory" "")

# include cmake modules as needed
include(ConfigVersion)
include(ConfigArchitecture)
include(ConfigCompilerFlags)
include(ConfigOMP)

# Math: find BLAS and LAPACK
set(BLAS_LANG   "CXX")
set(BLAS_FOUND FALSE)
set(LAPACK_LANG "CXX")
set(LAPACK_FOUND FALSE)
set(MKL_COMPILER_BINDINGS "${CMAKE_CXX_COMPILER_ID}")
include(ConfigMath)

include(ConfigMPI)
include(ConfigExplicitLibs)
include(ConfigSafeGuards)
include(GenericMacros)

init_FCMangle()

find_package(Threads REQUIRED)
link_libraries("${CMAKE_THREAD_LIBS_INIT}")

# Python Detection
include(ConfigPython)
link_libraries("${PYTHON_LIBRARIES}")
include_directories(SYSTEM "${PYTHON_INCLUDE_DIRS}")

# Boost Detection
# We need Boost.Python, so this has to come _after_ Python detection
include(ConfigBoost)
link_directories("${Boost_LIBRARY_DIRS}")
include_directories(SYSTEM "${Boost_INCLUDE_DIRS}")

# Append the suffix given from input to all generated executables
if(EXECUTABLE_SUFFIX)
   set(CMAKE_EXECUTABLE_SUFFIX "${EXECUTABLE_SUFFIX}")
   message(STATUS "Suffix ${CMAKE_EXECUTABLE_SUFFIX} will be appended to executables")
endif()

if(LDFLAGS)
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LDFLAGS}")
endif()

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

# library directory
add_subdirectory(lib)

# include directory (for installation)
add_subdirectory(include)

# recursively add source directories
add_subdirectory(src)

# test suite
add_subdirectory(test)

# sample suite
add_subdirectory(samples)

# Add all targets to the build-tree export set

# this has to be the very last CMake module to be included
include(ConfigInfo)
