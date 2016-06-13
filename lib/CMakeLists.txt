
if(ENABLE_PYTHON)
    install(DIRECTORY ambit DESTINATION ${PYTHON_SITE_PACKAGES} FILES_MATCHING PATTERN "*.py")
    install(FILES np_test.py test_blocks.py test_operators.py DESTINATION ${PYTHON_SITE_PACKAGES}/ambit/tests)
endif()
