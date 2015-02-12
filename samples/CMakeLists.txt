set(EXAMPLE1_SOURCES
    example1.cc
)

# build static libray
add_executable(example1 ${EXAMPLE1_SOURCES})
add_dependencies(example1 tensor)
target_link_libraries(example1 tensor ${Boost_LIBRARIES})

