set(PYTHON_HEADERS
)

set(PYTHON_SOURCES
    bindings.cc
)

# build static libray
add_library(ambit SHARED ${PYTHON_SOURCES} ${PYTHON_HEADERS})
add_dependencies(ambit tensor)

if(ENABLE_CYCLOPS)
    target_link_libraries(ambit ${CYCLOPS}/lib/libctf.a ${MPI_LIBRARIES})
    set_target_properties(ambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(ambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(ambit tensor ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#set_property(TARGET ambit PROPERTY OUTPUT_NAME "ambit.so")
set_target_properties(ambit PROPERTIES PREFIX "")
set_target_properties(ambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

#install(TARGETS ambit ARCHIVE DESTINATION lib)

