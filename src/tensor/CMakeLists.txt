set(TENSOR_HEADERS core.h tensorimpl.h)

set(TENSOR_SOURCES core.cc tensor.cc tensorimpl.cc)

# if we have MPI and Cyclops is enabled
#if(ENABLE_CYCLOPS)
#    list(APPEND TENSOR_HEADERS cyclops/cyclopsimpl.h)
#    list(APPEND TENSOR_SOURCES cyclops/cyclopsimpl.cc)
#endif()

# build static libray
add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})

# specify dependencies for the library (if any)
#add_dependencies(tensor )
install(TARGETS tensor ARCHIVE DESTINATION lib)
