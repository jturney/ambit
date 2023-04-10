set(TEST_OPERATORS_SOURCES
    test_operators.cc
)

# build static library
add_executable(test_operators ${TEST_OPERATORS_SOURCES})
target_link_libraries(test_operators ambit::ambit)
add_test(NAME operators COMMAND test_operators)

set(TEST_CORE_SOURCES
    test_core.cc
)

# build static library
add_executable(test_core ${TEST_CORE_SOURCES})
target_link_libraries(test_core ambit::ambit)
add_test(NAME core COMMAND test_core)

if(ENABLE_CYCLOPS)
    set(TEST_CYCLOPS_SOURCES
        test_cyclops.cc
    )

    add_executable(test_cyclops ${TEST_CYCLOPS_SOURCES})
    target_link_libraries(test_cyclops ambit-lib)
    add_test(NAME cyclops COMMAND test_cyclops)

    set(TEST_DISTRIBUTED_SOURCES
        test_distributed.cc
    )

    add_executable(test_distributed ${TEST_DISTRIBUTED_SOURCES})
    target_link_libraries(test_distributed ambit-lib)
    add_test(NAME distributed COMMAND test_distributed)
endif()

#set(TEST_HF_SOURCES
#    test_hf.cc
#)

# build static library
#add_executable(test_hf ${TEST_HF_SOURCES})
#target_link_libraries(test_hf ambit-lib)
#add_test(NAME hf COMMAND test_hf)

set(TEST_BLOCKS_SOURCES
    test_blocks.cc
)

# build static library
#add_executable(test_blocks ${TEST_BLOCKS_SOURCES})
#target_link_libraries(test_blocks ambit::ambit)
#add_test(NAME blocks COMMAND test_blocks)

set(TEST_SYM_BLOCKS_SOURCES
    test_sym_blocks.cc
)

# build static library
add_executable(test_sym_blocks ${TEST_SYM_BLOCKS_SOURCES})
target_link_libraries(test_sym_blocks ambit::ambit)
add_test(NAME sym_blocks COMMAND test_sym_blocks)

add_executable(test_performance test_performance.cc)
target_link_libraries(test_performance ambit::ambit)
add_test(NAME performance COMMAND test_performance)

add_executable(test_block_performance test_block_performance.cc)
target_link_libraries(test_block_performance ambit::ambit)
add_test(NAME block_performance COMMAND test_block_performance)

#configure_file(test.32 test.32 COPYONLY)
#configure_file(test.33 test.33 COPYONLY)
#configure_file(test.34 test.34 COPYONLY)
#configure_file(test.35 test.35 COPYONLY)

add_executable(test_hdf5 test_hdf5.cc)
target_link_libraries(test_hdf5 ambit::ambit)
add_test(NAME hdf5 COMMAND test_hdf5)

set(TEST_IO_SOURCES
    test_io.cc
)

# build static library
add_executable(test_io ${TEST_IO_SOURCES})
target_link_libraries(test_io ambit::ambit)
add_test(NAME io COMMAND test_io)
