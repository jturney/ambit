#if !defined(_AMBIT_GLOBALS_H_)
#define _AMBIT_GLOBALS_H_

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace ambit
{

// These are private globals to be use internally used by Ambit
namespace globals
{

#if defined(HAVE_MPI)

extern MPI_Comm communicator;

#endif
}
}

#endif //_AMBIT_GLOBALS_H_
