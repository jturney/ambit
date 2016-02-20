# Oldest version of Boost tested against
set(BOOSTVER 1.53.0)

list(APPEND needed_components filesystem system)

if(ENABLE_PYTHON)
    # On Macs with Boost Python installed with Homebrew with Python 3 the library
    # is named python3 (python is for Python 2). How this is handled on Linux
    # and other systems is in question.
    if(PYTHON_VERSION_MAJOR MATCHES "3")
        list(APPEND needed_components python3)
    elseif(PYTHON_VERSION_MAJOR MATCHES "2")
        list(APPEND needed_components python)
    else()
        message(FATAL "Unrecognized major version of Python: ${PYTHON_VERSION_MAJOR}")
    endif()
endif()

find_package(Boost ${BOOSTVER} REQUIRED COMPONENTS "${needed_components}")

