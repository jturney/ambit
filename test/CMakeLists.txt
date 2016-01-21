set(TEST_OPERATORS_SOURCES
    test_operators.cc
)

# build static libray
add_executable(test_operators ${TEST_OPERATORS_SOURCES})
add_dependencies(test_operators ambit)
target_link_libraries(test_operators ambit ${Boost_LIBRARIES})
add_test(NAME operators COMMAND test_operators)

set(TEST_CORE_SOURCES
    test_core.cc
)

# build static libray
add_executable(test_core ${TEST_CORE_SOURCES})
add_dependencies(test_core ambit)
target_link_libraries(test_core ambit ${Boost_LIBRARIES})
add_test(NAME core COMMAND test_core)

if(ENABLE_CYCLOPS)
    set(TEST_CYCLOPS_SOURCES
        test_cyclops.cc
    )

    add_executable(test_cyclops ${TEST_CYCLOPS_SOURCES})
    add_dependencies(test_cyclops ambit)
    target_link_libraries(test_cyclops ambit ${Boost_LIBRARIES})
    add_test(NAME cyclops COMMAND test_cyclops)

    set(TEST_DISTRIBUTED_SOURCES
        test_distributed.cc
    )

    add_executable(test_distributed ${TEST_DISTRIBUTED_SOURCES})
    add_dependencies(test_distributed ambit)
    target_link_libraries(test_distributed ambit ${Boost_LIBRARIES})
    add_test(NAME distributed COMMAND test_distributed)
endif()

set(TEST_HF_SOURCES
    test_hf.cc
)

# build static libray
add_executable(test_hf ${TEST_HF_SOURCES})
add_dependencies(test_hf ambit)
target_link_libraries(test_hf ambit ${Boost_LIBRARIES})
add_test(NAME hf COMMAND test_hf)

set(TEST_BLOCKS_SOURCES
    test_blocks.cc
)

# build static libray
add_executable(test_blocks ${TEST_BLOCKS_SOURCES})
add_dependencies(test_blocks ambit)
target_link_libraries(test_blocks ambit ${Boost_LIBRARIES})
add_test(NAME blocks COMMAND test_blocks)

add_executable(test_performance test_performance.cc)
add_dependencies(test_performance ambit)
target_link_libraries(test_performance ambit ${Boost_LIBRARIES})
add_test(NAME performance COMMAND test_performance)

configure_file(test.32 test.32 COPYONLY)
configure_file(test.33 test.33 COPYONLY)
configure_file(test.34 test.34 COPYONLY)
configure_file(test.35 test.35 COPYONLY)

add_executable(test_hdf5 test_hdf5.cc)
add_dependencies(test_hdf5 ambit)
target_link_libraries(test_hdf5 ambit ${Boost_LIBRARIES})
add_test(NAME hdf5 COMMAND test_hdf5)
