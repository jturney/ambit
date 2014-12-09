set(TENSOR_HEADERS )

set(TENSOR_SOURCES tensor.cc)

# build static libray
add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})

# specify dependencies for the library (if any)
#add_dependencies(tensor )
install(TARGETS tensor ARCHIVE DESTINATION lib)
