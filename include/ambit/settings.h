/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

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
