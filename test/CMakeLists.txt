set(TEST_OPERATORS_SOURCES
    test_operators.cc
)

# build static libray
add_executable(test_operators ${TEST_OPERATORS_SOURCES})
add_dependencies(test_operators tensor)
target_link_libraries(test_operators tensor ${Boost_LIBRARIES})
