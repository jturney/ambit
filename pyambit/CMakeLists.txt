if (CMAKE_HOST_WIN32)
    cmake_minimum_required(VERSION 3.21)
else()
    cmake_minimum_required(VERSION 3.16)
endif()
cmake_policy(SET CMP0079 NEW)

project(
  pyambit
  VERSION 0.6.0  # SYNC WITH `project(ambit VERSION <>)` in <top_level>/CMakeLists.txt
  )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

include(psi4OptionsTools)
option_with_default(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Release)
option_with_default(PYMOD_INSTALL_LIBDIR "Location within CMAKE_INSTALL_LIBDIR to which python modules are installed" /)
option_with_default(ambit_INSTALL_CMAKEDIR "Directory to which ambit CMake config files installed." share/cmake/ambit)

include(autocmake_safeguards)
set(CMAKECONFIG_INSTALL_DIR "${ambit_INSTALL_CMAKEDIR}")

if (NOT TARGET pybind11::pybind11)
    find_package(pybind11 CONFIG REQUIRED)
endif()

if (NOT TARGET Python::Module)
    set(Python_ADDITIONAL_VERSIONS 3.11 3.10 3.9 3.8 3.7)  # adjust with CMake minimum FindPythonInterp
    find_package(Python 3.7 COMPONENTS Interpreter Development NumPy REQUIRED)
    message(STATUS "Found Python ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}: ${Python_EXECUTABLE} (found version ${Python_VERSION})")
endif()

if (APPLE)
    set(base "@loader_path")
else()
    set(base "$ORIGIN")
endif()
file(RELATIVE_PATH relDir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit
                          ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH ${base} ${base}/${relDir})

message(STATUS "ambit rpath: ${CMAKE_INSTALL_RPATH}")

pybind11_add_module(pyambit MODULE bindings.cc)
target_compile_features(pyambit PRIVATE "cxx_std_17")
add_library(ambit::pyambit ALIAS pyambit)

if(NOT TARGET ambit::ambit)
    find_package(ambit CONFIG REQUIRED)
endif()
target_link_libraries(pyambit PRIVATE ambit::ambit)

# LAB: not addressed
if(ENABLE_CYCLOPS)
    target_link_libraries(pyambit PUBLIC ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

if(MSVC)
    target_compile_definitions(pyambit PUBLIC _USE_MATH_DEFINES)
    target_compile_options(pyambit PUBLIC "/EHsc")
    set_target_properties(
      pyambit
      PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
      )
endif()

set(export_properties
  "ambit_VERSION"
  "ambit_PYMOD"
  )

set_target_properties(
  pyambit
  PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    # below two are covered by pb11. Use pybind11_add_module(... NO_EXTRAS) to experiment here.
    # CXX_VISIBILITY_PRESET "hidden"
    # VISIBILITY_INLINES_HIDDEN 1
    ambit_VERSION ${pyambit_VERSION}
    ambit_PYMOD "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}"
    EXPORT_PROPERTIES "${export_properties}"
  )

# Should Python bindings get a <project>Config.cmake file for detecting them?
# * They form a python module, which has its own detection conventions, so maybe not.
# * They are a compiled CMake product, so maybe so.
# * Some have linkable symbols (e.g., Psi4) if the default hidden properties conferred by pb11 are countervened, so maybe so.
# * If one detects the C/C++ lib with CMake, it's nice to use the same scheme with the bindings, so maybe so.
# Should the Python bindings Config.cmake get a whole new project like <py_project>Config.cmake?
# * If pb11 module linked against static (or object) C/C++ lib (creating two fat largely duplicate libs), Py bindings are entirely indep, so maybe so.
# * If pb11 module linked against shared C/C++ lib (creating one fat, one lean lib), Py bindings are closely assoc. w/C/C++ project, so maybe not.
# The setup below answers the queries as "yes" and "no", respectively.
# * The bindings subproject installs "Targets-Python" exports detectable with `find_project(ambit COMPONENTS Python)`
# * Separating the C/C++ lib main project and bindings subproject is mainly for packagers looking to minimize build time for multiple Pythons.
# * Anyone not building C/C++ and Python together and who cares about install layout and ambit *consumption* should examine and test the find_package() circs.

install(
  TARGETS
    pyambit
  EXPORT
    python_interface
  LIBRARY DESTINATION
    ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit
  )

install(
  EXPORT
    python_interface
  FILE
    "ambitTargets-Python.cmake"
  NAMESPACE
    "ambit::"
  DESTINATION
    ${CMAKECONFIG_INSTALL_DIR}
  )

install(
  DIRECTORY ambit
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}
  FILES_MATCHING
    PATTERN "*.py"
  )
install(
  FILES
    np_test.py
    test_blocks.py
    test_operators.py
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit/tests
  )

