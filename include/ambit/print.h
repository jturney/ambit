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

#if !defined(TENSOR_INCLUDE_PRINT_H)
#define TENSOR_INCLUDE_PRINT_H

#include "common_types.h"

namespace ambit
{

/// Print function. Only the master node is allowed to print to the screen.
void print(const string format, ...);

/** Each process will print to their respective output file.
 *  The master process will print to both the screen and its output file.
 */
void printn(const string format, ...);

/** Increases printing column offset by increment.
 * @param increment the amount to increase indentation.
 */
void indent(int increment = 4);

/** Decreases printing column offset by increment.
 * @param decrement the amount the decrease indentation
 */
void unindent(int decrement = 4);

/** Returns the current level of indentation. */
int current_indent();

struct indenter
{
    indenter(int increment = 4) : size(increment) { indent(size); }
    ~indenter() { unindent(size); }

  private:
    int size;
};
}

#endif // TENSOR_INCLUDE_PRINT_H
