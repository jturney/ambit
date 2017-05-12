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

#include <ambit/print.h>
#include <ambit/settings.h>
#include <ambit/tensor.h>

#include <cstdarg>

namespace ambit
{

namespace
{

int indent_size = 0;

void print_indentation() { printf("%*s", indent_size, ""); }
}

int current_indent() { return indent_size; }

void indent(int increment) { indent_size += increment; }

void unindent(int decrement)
{
    indent_size -= decrement;
    if (indent_size < 0)
        indent_size = 0;
}

void print(const string &format, ...)
{
    if (ambit::settings::rank == 0)
    {
        va_list args;
        va_start(args, format);
        print_indentation();
        vprintf(format.c_str(), args);
        va_end(args);
    }
}

void printn(const string &format, ...)
{
    for (int proc = 0; proc < settings::nprocess; proc++)
    {
        if (proc == settings::rank)
        {
            printf("%d: ", settings::rank);
            va_list args;
            va_start(args, format);
            print_indentation();
            vprintf(format.c_str(), args);
            va_end(args);
        }

        barrier();
    }
}
}
