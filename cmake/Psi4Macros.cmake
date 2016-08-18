###This file contains functions used throughout the Psi4 build.  Like source
###code, the build system should be factored and common code extracted out into
###functions/macros.  If you find repetitive code throughout the build scripts
###this is the place to add it (make sure you document it too).

#Macro for printing an option in a consistent manner
#
#Syntax: print_option(<option to print> <was specified>)
#
macro(print_option variable default)
if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
    message(STATUS "Setting (unspecified) option ${variable}: ${default}")
else()
    message(STATUS "Setting option ${variable}: ${${variable}}")
endif()
endmacro()

# Wraps an option with default ON/OFF. Adds nice messaging to option()
#
#Syntax: option_with_print(<option name> <description> <default value>)
#
macro(option_with_print variable msge default)
   print_option(${variable} ${default})
   option(${variable} ${msge} ${default})
endmacro(option_with_print)

#Wraps an option with a default other than ON/OFF and prints it
#NOTE: Can't combine with above b/c CMake handles ON/OFF options specially
#NOTE2: CMAKE_BUILD_TYPE (and other CMake variables) are always defined so need
#       to further check for if they are the NULL string.  This is also why we
#       need the force
#
#Syntax: option_with_default(<option name> <description> <default value>)
#
macro(option_with_default variable msge default)
print_option(${variable} ${default})
if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
   set(${variable} ${default} CACHE STRING ${msge} FORCE)
endif()
endmacro(option_with_default)

#Common guts to adding a Psi4 library irrespective of bin vs. lib home
#NOTE: list of sources is a CMake list
#
#Syntax general_add_library(<library name>, <list of sources>, <lib or bin>,
#                           <dependencies>)
#
macro(general_add_library libname sources dir)
   #TODO: Switch to OBJECT library?  Simplifies this macro...
   if(${dir} MATCHES lib)
      set(prefix lib)
   endif()
   add_library(${libname} ${${sources}})
   set_target_properties(${libname} PROPERTIES 
       POSITION_INDEPENDENT_CODE ${BUILD_FPIC}
   )
   install(TARGETS ${libname} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
   install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
      DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/psi4/src/${dir}/${prefix}${libname}
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
   set_property(GLOBAL APPEND PROPERTY LIBLIST ${libname})
   set(depend_name "${ARGN}")
   foreach(name_i IN LISTS depend_name)
      target_link_libraries(${libname} INTERFACE ${name_i})
   endforeach()
   target_include_directories(${libname} PUBLIC ${Boost_INCLUDE_DIRS} 
                                                ${LIBDERIV_INCLUDE_DIRS})
endmacro(general_add_library libname sources prefix dir)

#Adds a psi4 library that lives in lib
#
#Syntax: psi4_add_library(<library name> <list of sources> <dependencies>)
#
macro(psi4_add_library libname sources)
   general_add_library(${libname} ${sources} lib ${ARGN}) 
endmacro(psi4_add_library libname sources)

#Adds a psi4 library that lives in bin
#
#Syntax: psi4_add_binary(<library name> <list of sources> <dependencies>
#
macro(psi4_add_binary libname sources)
   general_add_library(${libname} ${sources} bin ${ARGN})
endmacro(psi4_add_binary libname sources)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckFortranCompilerFlag)  # CMake >= 3.3, so local copy in cmake/

#The guts of the next two functions, use the wrappers please
#
#Syntax: add_C_or_CXX_flags(<True for C, False for CXX>)
#
# Note: resist adding -Werror to the check_X_compiler_flag calls,
#   as (i) the flag for Intel is actually -diag-error warn, (ii)
#   Intel ifort doesn't define -Werror, and (iii) passing it
#   changes REQUIRED_DEFINITIONS.
macro(add_C_or_CXX_flags is_C)
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
   set(CMAKE_REQUIRED_QUIET ON)
   set(flags_to_try "${ARGN}")
   foreach(flag_i IN LISTS flags_to_try ITEMS -brillig)
      if(${flag_i} STREQUAL "-brillig")
         message(WARNING "Option unfulfilled as none of ${flags_to_try} valid")
         break()
      endif()
      unset(test_option CACHE)
      if(${is_C} EQUAL 0)
          CHECK_C_COMPILER_FLAG("${flag_i}" test_option)
          set(description_to_print CMAKE_C_FLAGS)
      elseif(${is_C} EQUAL 1)
          CHECK_CXX_COMPILER_FLAG("${flag_i}" test_option)
          set(description_to_print CMAKE_CXX_FLAGS)
      elseif(${is_C} EQUAL 2)
          CHECK_Fortran_COMPILER_FLAG("${flag_i}" test_option)
          set(description_to_print CMAKE_Fortran_FLAGS)
      endif()
      set(msg_base "Performing Test ${description_to_print} [${flag_i}] -")
      if(${test_option})
        set(${description_to_print} "${${description_to_print}} ${flag_i}")
        if(NOT CMAKE_REQUIRED_QUIET_SAVE)
           message(STATUS  "${msg_base} Success, Appending")
        endif()
        break()
      else()
        if(NOT CMAKE_REQUIRED_QUIET_SAVE)
           message(STATUS "${msg_base} Failed")
        endif()
      endif()
   endforeach()
   set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})  
endmacro()



#Checks if C flags are valid, if so adds them to CMAKE_C_FLAGS
#Input should be a list of flags to try.  If two flags are to be tried together
#enclose them in quotes, e.g. "-L/path/to/dir -lmylib" is tried as a single
#flag, whereas "-L/path/to/dir" "-lmylib" is tried as two separate flags.
#The first list item to succeed is added to CMAKE_C_FLAGS, then try loop
#breaks. Warning issued if no flags in list succeed.
#
#
#Syntax: add_C_flags(<flags to add>)
#
macro(add_C_flags)
   add_C_or_CXX_flags(0 ${ARGN})
endmacro()

#Checks if CXX flags are valid, if so adds them to CMAKE_CXX_FLAGS
#See add_C_flags for more info on syntax
#
#Syntax: add_CXX_flags(<flags to add>)
#
macro(add_CXX_flags)
    add_C_or_CXX_flags(1 ${ARGN})
endmacro()

#Checks if Fortran flags are valid, if so adds them to CMAKE_Fortran_FLAGS
#See add_C_flags for more info on syntax
#
#Syntax: add_Fortran_flags(<flags to add>)
#
macro(add_Fortran_flags)
    add_C_or_CXX_flags(2 ${ARGN})
endmacro()

#Macro for adding flags common to both C and CXX, if the compiler supports them
#
#Syntax: add_flags(<flags to add>)
#
macro(add_flags FLAGS)
    if(CMAKE_C_COMPILER)
        add_C_flags(${FLAGS})
    endif()
    if(CMAKE_CXX_COMPILER)
        add_CXX_flags(${FLAGS})
    endif()
    if(CMAKE_Fortran_COMPILER)
        add_Fortran_flags(${FLAGS})
    endif()
endmacro()

#Defines an option that if enabled turns on some compiler flags
#
#Syntax: option_with_flags(<option> <description> <default value> <flags>)
#
macro(option_with_flags option msg default)
    print_option(${option} ${default})
    option(${option} ${msg} ${default})
    if(${${option}})
       add_flags("${ARGN}")
    endif()
endmacro()

#Macro so I don't have to look at a ton of if statements for adding each plugin
#
#Syntax: optional_plugin(<plugin name>)
#
macro(optional_plugin plugin_name)
string(TOUPPER ${plugin_name} PLUGIN_NAME)
if(${ENABLE_${PLUGIN_NAME}})
   find_package(${plugin_name} REQUIRED)
   set_property(GLOBAL APPEND PROPERTY PSI4_MODULES ${${PLUGIN_NAME}_LIBRARIES})
   add_definitions(-DENABLE_${PLUGIN_NAME})
else()
   add_library(${plugin_name} INTERFACE)
endif()
endmacro(optional_plugin plugin_name)

# Generate the FCMangle header and post-process it to add the copyright notice
macro(init_FCMangle)
    get_fc_symbol(FCSYMBOL)
    if(FCSYMBOL EQUAL 1) # lower case
        set(contents
                "
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name,NAME) name

/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL_(name,NAME) name
")
    elseif(FCSYMBOL EQUAL 2) #lower case underscore
        set(contents
                "
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name,NAME) name##_

/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL_(name,NAME) name##_
")
    elseif(FCSYMBOL EQUAL 3) #upper case
        set(contents
                "
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name,NAME) NAME

/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL_(name,NAME) NAME
")
    else() #upper case underscore
        set(contents
                "
/* Mangling for Fortran global symbols without underscores. */
#define FC_GLOBAL(name,NAME) NAME##_

/* Mangling for Fortran global symbols with underscores. */
#define FC_GLOBAL_(name,NAME) NAME##_
")
    endif()
    file(WRITE ${PROJECT_BINARY_DIR}/include/FCMangle.h
            "/*\n *@BEGIN LICENSE\n *@END LICENSE\n */\n\n")
    file(APPEND ${PROJECT_BINARY_DIR}/include/FCMangle.h "// Header file automagically generated by CMake. DO NOT TOUCH!\n")
    file(APPEND ${PROJECT_BINARY_DIR}/include/FCMangle.h "#if !defined(FC_HEADER_INCLUDED)\n#define FC_HEADER_INCLUDED\n")
    file(APPEND ${PROJECT_BINARY_DIR}/include/FCMangle.h ${contents})
    file(APPEND ${PROJECT_BINARY_DIR}/include/FCMangle.h "\n#endif\n")
endmacro()

function(test_fortran_mangling PREFIX ISUPPER POSTFIX FLAGS SUB RESULT)
    if(ISUPPER)
        string(TOUPPER "${SUB}" sub)
    else(ISUPPER)
        string(TOLOWER "${SUB}" sub)
    endif(ISUPPER)
    set(FUNCTION "${PREFIX}${sub}${POSTFIX}")
    # create a fortran file with sub called sub
    set(TMP_DIR
            "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/CheckFortranLink")
    file(REMOVE_RECURSE "${TMP_DIR}")
    message(STATUS "checking Fortran linkage: ${FUNCTION}")
    file(WRITE "${TMP_DIR}/ctof.c"
            "
      extern ${FUNCTION}();
      int main() { ${FUNCTION}(); return 0;}
    "
            )
    file(WRITE "${TMP_DIR}/CMakeLists.txt"
            "
     project(testf C)
     set(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} ${FLAGS}\")
     add_executable(ctof ctof.c)
     target_link_libraries(ctof ${BLAS_LIBRARIES})
    "
            )
    set(FORTRAN_NAME_MANGLE_TEST FALSE)
    try_compile(FORTRAN_NAME_MANGLE_TEST "${TMP_DIR}" "${TMP_DIR}"
            testf
            OUTPUT_VARIABLE output)
    #if(output)
    #  message(${output})
    #  endif()
    if(FORTRAN_NAME_MANGLE_TEST)
        set(${RESULT} TRUE PARENT_SCOPE)
    else()
        set(${RESULT} FALSE PARENT_SCOPE)
    endif()
endfunction(test_fortran_mangling)

function(get_fc_symbol FCSYMBOLOUT)
    #test_fortran_mangling(pre isUpper post flags      sub   worked )
    test_fortran_mangling( ""  True    "_"  ""         "DGEMM" FC_LINK_WORKED)
    if(FC_LINK_WORKED)
        set(${FCSYMBOLOUT} 4 PARENT_SCOPE)
        message(STATUS "Upper case with underscore is used")
        return()
    endif()
    test_fortran_mangling("" False   "_" "" "dgemm" FC_LINK_WORKED)
    if(FC_LINK_WORKED)
        set(${FCSYMBOLOUT} 2 PARENT_SCOPE)
        message(STATUS "Lower case with underscore is used")
        return()
    endif()
    test_fortran_mangling("" True    "" "" "DGEMM" FC_LINK_WORKED)
    if(FC_LINK_WORKED)
        set(${FCSYMBOLOUT} 3 PARENT_SCOPE)
        message(STATUS "Upper case (no underscore) is used")
        return()
    endif()
    test_fortran_mangling("" False   "" "" "dgemm" FC_LINK_WORKED)
    if(FC_LINK_WORKED)
        set(${FCSYMBOLOUT} 1 PARENT_SCOPE)
        message(STATUS "Lower case (no underscore) is used")
        return()
    endif()
    message(FATAL_ERROR "Unable to detect Fortran name mangling. This should not happen, please set --with-f77symbol manually.")
endfunction(get_fc_symbol)
