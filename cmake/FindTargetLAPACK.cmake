# FindTargetLAPACK.cmake
# --------------------
#
# LAPACK cmake module to wrap FindLAPACK.cmake in a target.
#
# This module sets the following variables in your project: ::
#
#   TargetLAPACK_FOUND - true if BLAS/LAPACK found on the system
#   TargetLAPACK_MESSAGE - status message with BLAS/LAPACK library path list
#
# This module *unsets* the following conventional LAPACK variables so as
#   to force using the target: ::
#
#   BLAS_FOUND
#   BLAS_LIBRARIES
#   LAPACK_FOUND
#   LAPACK_LIBRARIES
#

set(PN TargetLAPACK)

# 1st precedence - libraries passed in through -DLAPACK_LIBRARIES
if (LAPACK_LIBRARIES)
    if (NOT ${PN}_FIND_QUIETLY)
        message (STATUS "LAPACK detection suppressed.")
    endif()

    add_library (tgt::blas INTERFACE IMPORTED)
    add_library (tgt::lapack INTERFACE IMPORTED)
    set_property (TARGET tgt::blas PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
    set_property (TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
else()
    # 2nd precedence - target already prepared and findable in TargetLAPACKConfig.cmake
    find_package (TargetLAPACK QUIET CONFIG)
    if (TARGET tgt::lapack)
        if (NOT ${PN}_FIND_QUIETLY)
            message (STATUS "TargetLAPACKConfig detected.")
        endif()
    else()
        # 3rd precedence - usual variables from FindLAPACK.cmake or ConfigMath
        include(ConfigMath)
        if (NOT ${PN}_FIND_QUIETLY)
            message (STATUS "ConfigMath LAPACK detected.")
        endif()
    
        add_library (tgt::blas INTERFACE IMPORTED)
        add_library (tgt::lapk INTERFACE IMPORTED)
        add_library (tgt::lapack INTERFACE IMPORTED)
        set_property (TARGET tgt::blas PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})
        set_property (TARGET tgt::lapk PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
        set_property (TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES lapk blas)

        unset (BLAS_LIBRARIES)
        unset (LAPACK_LIBRARIES)
    endif()
endif()    

get_property (_ill TARGET tgt::lapack PROPERTY INTERFACE_LINK_LIBRARIES)
set (${PN}_MESSAGE "Found LAPACK: ${_ill}")
if ((TARGET tgt::blas) AND (TARGET tgt::lapk))
    get_property (_illb TARGET tgt::blas PROPERTY INTERFACE_LINK_LIBRARIES)
    get_property (_illl TARGET tgt::lapk PROPERTY INTERFACE_LINK_LIBRARIES)
    set (${PN}_MESSAGE "Found LAPACK: ${_illl};${_illb}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args (${PN} DEFAULT_MSG ${PN}_MESSAGE)
