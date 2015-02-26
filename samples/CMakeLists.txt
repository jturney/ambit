set(EXAMPLE1_SOURCES
    example1.cc
)

set(EXAMPLE2_SOURCES
    example2.cc
)

# build static libray
add_executable(example1 ${EXAMPLE1_SOURCES})
add_executable(example2 ${EXAMPLE2_SOURCES})
add_dependencies(example1 ambit)
add_dependencies(example2 ambit)
target_link_libraries(example1 ambit ${Boost_LIBRARIES})
target_link_libraries(example2 ambit ${Boost_LIBRARIES})

