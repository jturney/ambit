set(EXAMPLE1_SOURCES
    example1.cc
)

set(EXAMPLE2_SOURCES
    example2.cc
)

# build static libray
add_executable(example1 ${EXAMPLE1_SOURCES})
add_executable(example2 ${EXAMPLE2_SOURCES})
target_link_libraries(example1 ambit-lib)
target_link_libraries(example2 ambit-lib)

