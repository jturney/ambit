set(TEST_OPERATORS_SOURCES
    test_operators.cc
)

# build static libray
add_executable(test_operators ${TEST_OPERATORS_SOURCES})
add_dependencies(test_operators ambit)
target_link_libraries(test_operators ambit ${Boost_LIBRARIES})

set(TEST_CORE_SOURCES
    test_core.cc
)

# build static libray
add_executable(test_core ${TEST_CORE_SOURCES})
add_dependencies(test_core ambit)
target_link_libraries(test_core ambit ${Boost_LIBRARIES})

if(ENABLE_CYCLOPS)
    set(TEST_CYCLOPS_SOURCES
        test_cyclops.cc
    )

    add_executable(test_cyclops ${TEST_CYCLOPS_SOURCES})
    add_dependencies(test_cyclops ambit)
    target_link_libraries(test_cyclops ambit ${Boost_LIBRARIES})

    set(TEST_DISTRIBUTED_SOURCES
        test_distributed.cc
    )

    add_executable(test_distributed ${TEST_DISTRIBUTED_SOURCES})
    add_dependencies(test_distributed ambit)
    target_link_libraries(test_distributed ambit ${Boost_LIBRARIES})
endif()

set(TEST_HF_SOURCES
    test_hf.cc
)

# build static libray
add_executable(test_hf ${TEST_HF_SOURCES})
add_dependencies(test_hf ambit)
target_link_libraries(test_hf ambit ${Boost_LIBRARIES})

set(TEST_BLOCKS_SOURCES
    test_blocks.cc
)

# build static libray
add_executable(test_blocks ${TEST_BLOCKS_SOURCES})
add_dependencies(test_blocks ambit)
target_link_libraries(test_blocks ambit ${Boost_LIBRARIES})

add_executable(test_performance test_performance.cc)
add_dependencies(test_performance ambit)
target_link_libraries(test_performance ambit ${Boost_LIBRARIES})

configure_file(test.32 test.32 COPYONLY)
configure_file(test.33 test.33 COPYONLY)
configure_file(test.34 test.34 COPYONLY)
configure_file(test.35 test.35 COPYONLY)

