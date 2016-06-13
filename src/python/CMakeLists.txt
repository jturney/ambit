if (NOT STATIC_ONLY)

set(PYTHON_HEADERS
)

set(PYTHON_SOURCES
    bindings.cc
)

# build static libray
add_library(pyambit SHARED ${PYTHON_SOURCES} ${PYTHON_HEADERS})
add_dependencies(pyambit ambit)

if(ENABLE_CYCLOPS)
    target_link_libraries(pyambit ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(pyambit ambit ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# Ensure the Python module is appropriately named
set_target_properties(pyambit PROPERTIES PREFIX "")
set_target_properties(pyambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

install(TARGETS pyambit DESTINATION ${PYTHON_SITE_PACKAGES}/ambit)

endif()
