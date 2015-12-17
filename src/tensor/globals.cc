#include "globals.h"

namespace ambit
{

// These are private globals to be use internally used by Ambit
namespace globals
{

#if defined(HAVE_MPI)

MPI_Comm communicator = 0;

#endif
}
}
