//
// Created by Justin Turney on 10/20/15.
//

#ifndef AMBIT_SETTINGS_H
#define AMBIT_SETTINGS_H

namespace ambit
{

// => Settings Namespace <=
namespace settings
{

/** Number of MPI processes.
 *
 * For single process runs this will always be 1.
 */
extern int nprocess;

/// Rank of this process. (zero-based)
extern int rank;

/// Print debug information? true, or false
extern bool debug;

/// Memory usage limit. Default is 1GB.
extern size_t memory_limit;

/// Distributed capable?
extern const bool distributed_capable;

/// Enable timers
extern bool timers;
}
}

#endif // AMBIT_SETTINGS_H
