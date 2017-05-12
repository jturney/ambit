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

#if !defined(TENSOR_HELPERS_PSI4_IO_H)
#define TENSOR_HELPERS_PSI4_IO_H

#include <ambit/io/psi4/io.h>

namespace ambit
{
namespace helpers
{
namespace psi4
{

/** Loads a matrix from a Psi4 data file.
 *
 * When called in an MPI run, the master node performs the read operation
 * and broadcasts the data as needed via the Tensor mechanics.
 *
 * @param fn The filename to read from.
 * @param entry The TOC entry in the Psi4 file to load.
 * @param target The target tensor to place the data.
 */
void load_matrix(const std::string &fn, const std::string &entry,
                 Tensor &target);

/** Loads two-electron integrals from an IWL file into the tensor.
*
* @param fn The filename to read from.
* @param target The tensor to place data into.
*/
void load_iwl(const std::string &fn, Tensor &target);
}
}
}

#endif
