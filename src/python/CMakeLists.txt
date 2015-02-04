set(PYTHON_HEADERS
)

set(PYTHON_SOURCES
    bindings.cc
)

# build static libray
add_library(pytensor SHARED ${PYTHON_SOURCES} ${PYTHON_HEADERS})
add_dependencies(pytensor tensor)

if(ENABLE_CYCLOPS)
    target_link_libraries(pytensor ${CYCLOPS}/lib/libctf.a ${MPI_LIBRARIES})
    set_target_properties(pytensor PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pytensor PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(pytensor tensor ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#set_property(TARGET pytensor PROPERTY OUTPUT_NAME "pytensor.so")
set_target_properties(pytensor PROPERTIES PREFIX "")
set_target_properties(pytensor PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

#install(TARGETS pytensor ARCHIVE DESTINATION lib)

