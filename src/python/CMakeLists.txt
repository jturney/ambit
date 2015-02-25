set(PYTHON_HEADERS
)

set(PYTHON_SOURCES
    bindings.cc
)

# build static libray
add_library(pyambit SHARED ${PYTHON_SOURCES} ${PYTHON_HEADERS})
add_dependencies(pyambit ambit)

if(ENABLE_CYCLOPS)
    target_link_libraries(pyambit ${CYCLOPS}/lib/libctf.a ${MPI_LIBRARIES})
    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(pyambit ambit ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#set_property(TARGET ambit PROPERTY OUTPUT_NAME "ambit.so")
set_target_properties(pyambit PROPERTIES PREFIX "")
set_target_properties(pyambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

#install(TARGETS ambit ARCHIVE DESTINATION lib)

